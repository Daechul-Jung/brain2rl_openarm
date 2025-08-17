#!/usr/bin/env python3
import argparse
import numpy as np

from brain2rl_openarm.envs.openarm_env import OpenArmEnv

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=3)
    p.add_argument('--steps', type=int, default=300)
    args = p.parse_args()

    env = OpenArmEnv(use_velocity=False, action_scale=0.03, horizon=args.steps)
    returns = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        ep_ret = 0.0
        for t in range(args.steps):
            action = env.action_space.sample()
            obs, r, term, trunc, info = env.step(action)
            ep_ret += r
            if term or trunc:
                break
        returns.append(ep_ret)
        print(f"[rl_train_joint] ep {ep+1}/{args.episodes} return = {ep_ret:.3f}")
    env.close()
    print(f"[rl_train_joint] mean return: {np.mean(returns):.3f}")

if __name__ == "__main__":
    main()
