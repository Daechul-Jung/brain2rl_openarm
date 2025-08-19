#!/usr/bin/env python3
"""
Script to run OpenArm environment with brain2rl agents.
This script demonstrates how to integrate the OpenArm environment with your brain2rl framework.
"""

import argparse
import sys
import os
import numpy as np

# Add the brain2rl directory to the Python path
brain2rl_path = os.path.expanduser('~/brain2rl')
if brain2rl_path not in sys.path:
    sys.path.insert(0, brain2rl_path)

try:
    from brain2rl_openarm.envs.openarm_env import OpenArmEnv
    print("[run_with_agent] Successfully imported OpenArmEnv")
except ImportError as e:
    print(f"Error importing OpenArmEnv: {e}")
    print("Make sure the brain2rl_openarm package is properly built and sourced")
    sys.exit(1)

def create_random_agent(env):
    """Create a simple random agent for testing"""
    class RandomAgent:
        def __init__(self, action_space):
            self.action_space = action_space
        
        def act(self, obs):
            return self.action_space.sample()
        
        def reset(self):
            pass
    
    return RandomAgent(env.action_space)

def create_ppo_agent(env, model_path):
    """Create a PPO agent from brain2rl (placeholder)"""
    try:
        # This is a placeholder - you'll need to implement based on your brain2rl framework
        print(f"[run_with_agent] Attempting to load PPO agent from {model_path}")
        
        # Placeholder for PPO agent creation
        # You'll need to implement this based on your brain2rl framework
        print("[run_with_agent] PPO agent loading not yet implemented")
        print("[run_with_agent] Falling back to random agent")
        return create_random_agent(env)
        
    except Exception as e:
        print(f"[run_with_agent] Error loading PPO agent: {e}")
        print("[run_with_agent] Falling back to random agent")
        return create_random_agent(env)

def main():
    p = argparse.ArgumentParser(description='Run OpenArm environment with brain2rl agent')
    p.add_argument('--episodes', type=int, default=5, help='Number of evaluation episodes')
    p.add_argument('--steps', type=int, default=300, help='Maximum steps per episode')
    p.add_argument('--action-scale', type=float, default=0.03, help='Action scaling factor')
    p.add_argument('--use-velocity', action='store_true', help='Include velocity in observations')
    p.add_argument('--agent-type', choices=['random', 'ppo'], default='random', help='Type of agent to use')
    p.add_argument('--model-path', type=str, default='~/brain2rl/ppo_humanoid.pth', help='Path to trained model (for PPO)')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    args = p.parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    print(f"[run_with_agent] Starting evaluation with {args.episodes} episodes")
    print(f"[run_with_agent] Agent type: {args.agent_type}")
    print(f"[run_with_agent] Action scale: {args.action_scale}")
    
    try:
        # Create environment
        env = OpenArmEnv(
            use_velocity=args.use_velocity,
            action_scale=args.action_scale,
            horizon=args.steps
        )
        print(f"[run_with_agent] Environment created successfully")
        
        # Create agent
        if args.agent_type == 'ppo':
            model_path = os.path.expanduser(args.model_path)
            agent = create_ppo_agent(env, model_path)
        else:
            agent = create_random_agent(env)
        
        print(f"[run_with_agent] Agent created: {type(agent).__name__}")
        
        # Evaluation loop
        returns = []
        for ep in range(args.episodes):
            obs, _ = env.reset()
            agent.reset()
            ep_ret = 0.0
            print(f"[run_with_agent] Episode {ep+1}/{args.episodes} started")
            
            for t in range(args.steps):
                # Get action from agent
                action = agent.act(obs)
                
                # Take step in environment
                obs, r, term, trunc, info = env.step(action)
                ep_ret += r
                
                if t % 50 == 0:
                    print(f"  Step {t}: reward={r:.3f}, cumulative={ep_ret:.3f}")
                
                if term or trunc:
                    break
            
            returns.append(ep_ret)
            print(f"[run_with_agent] Episode {ep+1}/{args.episodes} finished: return = {ep_ret:.3f}")
        
        # Print evaluation summary
        print(f"\n[run_with_agent] Evaluation completed!")
        print(f"[run_with_agent] Mean return: {np.mean(returns):.3f}")
        print(f"[run_with_agent] Std return: {np.std(returns):.3f}")
        print(f"[run_with_agent] Min return: {np.min(returns):.3f}")
        print(f"[run_with_agent] Max return: {np.max(returns):.3f}")
        
    except Exception as e:
        print(f"[run_with_agent] Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        try:
            env.close()
            print("[run_with_agent] Environment closed successfully")
        except Exception as e:
            print(f"[run_with_agent] Warning: Error closing environment: {e}")

if __name__ == "__main__":
    main()
