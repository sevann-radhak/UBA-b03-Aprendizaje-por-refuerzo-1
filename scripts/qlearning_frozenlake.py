"""
Q-learning on Gymnasium FrozenLake-v1.
Saves per-episode rewards, moving-average plot, and Q-table for evaluation.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Q-learning on FrozenLake-v1")
    p.add_argument("--episodes", type=int, default=20_000)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-min", type=float, default=0.05)
    p.add_argument("--epsilon-decay", type=float, default=0.9995)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-slippery", action="store_true", help="Deterministic transitions")
    p.add_argument("--tag", type=str, default="slippery", help="Suffix for output filenames")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "informe" / "assets",
    )
    return p.parse_args()


def train_qlearning(
    env: gym.Env,
    episodes: int,
    alpha: float,
    gamma: float,
    eps_start: float,
    eps_min: float,
    eps_decay: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[float]]:
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q = np.zeros((n_states, n_actions), dtype=np.float64)
    rewards: list[float] = []
    epsilon = eps_start

    for _ in range(episodes):
        state, _ = env.reset(seed=int(rng.integers(1 << 30)))
        total_r = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            if rng.random() < epsilon:
                action = int(rng.integers(n_actions))
            else:
                action = int(np.argmax(q[state]))

            next_state, reward, terminated, truncated, _ = env.step(action)
            best_next = float(np.max(q[next_state]))
            td_target = reward + (0.0 if terminated else gamma * best_next)
            q[state, action] += alpha * (td_target - q[state, action])
            total_r += reward
            state = next_state

        rewards.append(total_r)
        epsilon = max(eps_min, epsilon * eps_decay)

    return q, rewards


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    x = x.astype(np.float64)
    if window <= 1:
        return x
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(x, kernel, mode="same")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    is_slippery = not args.no_slippery
    env = gym.make("FrozenLake-v1", is_slippery=is_slippery)

    q, rewards = train_qlearning(
        env,
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        eps_start=args.epsilon_start,
        eps_min=args.epsilon_min,
        eps_decay=args.epsilon_decay,
        rng=rng,
    )
    env.close()

    rew = np.array(rewards, dtype=np.float64)
    ma = moving_average(rew, window=100)

    csv_path = args.out_dir / f"rewards_{args.tag}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode", "reward"])
        for i, r in enumerate(rewards):
            w.writerow([i, r])

    fig_path = args.out_dir / f"convergence_{args.tag}.png"
    plt.figure(figsize=(9, 4))
    plt.plot(rew, alpha=0.25, label="Recompensa por episodio")
    plt.plot(ma, linewidth=1.5, label="Media móvil (100 episodios)")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa")
    plt.title(f"FrozenLake-v1 — Q-learning ({'resbaladizo' if is_slippery else 'determinístico'})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    q_path = args.out_dir / f"qtable_{args.tag}.npy"
    np.save(q_path, q)

    print(f"Wrote {csv_path}")
    print(f"Wrote {fig_path}")
    print(f"Wrote {q_path}")
    print(f"Mean reward last 500 episodes: {rew[-500:].mean():.4f}")


if __name__ == "__main__":
    main()
