#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np

from brain2rl_openarm.envs.openarm_env import OpenArmEnv

HOME = os.path.expanduser("~")
B2R = os.path.join(HOME, "brain2rl")
if B2R not in sys.path:
    sys.path.insert(0, B2R)
RL_DIR = os.path.join(B2R, "models", "rl")
if RL_DIR not in sys.path:
    sys.path.insert(0, RL_DIR)
for p in (B2R, RL_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
        
sys.path.insert(0, os.path.join(B2R, "practice", "agents"))
from rl_agents_prac import PPOAgent, SACAgent


ALGOS = {"ppo": PPOAgent, "sac": SACAgent}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--algo', choices=list(ALGOS.keys()), default='ppo')
    p.add_argument('--episodes', type=int, default=3)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--ckpt', type=str, default=None, help='path to checkpoint')
    args = p.parse_args()

    env = OpenArmEnv(action_scale=0.03, horizon=args.steps)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    AgentCls = ALGOS[args.algo]
    if AgentCls is None:
        print("[run_with_agent] No agent available; falling back to random actions.")
        agent = None
    else:
        agent = AgentCls(obs_dim, act_dim)
        if args.ckpt and hasattr(agent, "load"):
            agent.load(args.ckpt)

    for ep in range(args.episodes):
        obs, _ = env.reset()
        return_value = 0.0
        for t in range(args.steps):
            if agent is None or not hasattr(agent, "act"):
                action = env.action_space.sample()
            else:
                action = agent.act(obs)  # expected to return np.ndarray
            obs, r, term, trunc, _ = env.step(action)
            return_value += r
            if term or trunc:
                break
        print(f"[run_with_agent] ep {ep+1} return={return_value:.3f}")
    env.close()

if __name__ == "__main__":
    main()
