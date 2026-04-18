"""Evaluate greedy policy from a saved Q-table on FrozenLake-v1."""

from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate greedy Q-table policy")
    p.add_argument("--q-table", type=str, required=True)
    p.add_argument("--episodes", type=int, default=2_000)
    p.add_argument("--no-slippery", action="store_true")
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    q = np.load(args.q_table)
    is_slippery = not args.no_slippery
    env = gym.make("FrozenLake-v1", is_slippery=is_slippery)
    successes = 0
    rng = np.random.default_rng(args.seed)

    for ep in range(args.episodes):
        state, _ = env.reset(seed=int(rng.integers(1 << 30)))
        terminated = truncated = False
        while not (terminated or truncated):
            action = int(np.argmax(q[state]))
            state, reward, terminated, truncated, _ = env.step(action)
        if reward > 0:
            successes += 1

    env.close()
    rate = successes / args.episodes
    print(f"Success rate: {rate:.3f} ({successes}/{args.episodes})")


if __name__ == "__main__":
    main()
