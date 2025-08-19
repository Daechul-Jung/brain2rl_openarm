#!/usr/bin/env python3
import argparse
import numpy as np
import sys
import os

# Add the brain2rl directory to the Python path
brain2rl_path = os.path.expanduser('~/brain2rl')
if brain2rl_path not in sys.path:
    sys.path.insert(0, brain2rl_path)

try:
    from brain2rl_openarm.envs.openarm_env import OpenArmEnv
except ImportError as e:
    print(f"Error importing OpenArmEnv: {e}")
    print("Make sure the brain2rl_openarm package is properly built and sourced")
    sys.exit(1)

def main():
    p = argparse.ArgumentParser(description='Train RL agent on OpenArm environment')
    p.add_argument('--episodes', type=int, default=3, help='Number of training episodes')
    p.add_argument('--steps', type=int, default=300, help='Maximum steps per episode')
    p.add_argument('--action-scale', type=float, default=0.03, help='Action scaling factor')
    p.add_argument('--use-velocity', action='store_true', help='Include velocity in observations')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    args = p.parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    print(f"[rl_train_joint] Starting training with {args.episodes} episodes, {args.steps} steps per episode")
    print(f"[rl_train_joint] Action scale: {args.action_scale}, Use velocity: {args.use_velocity}")
    
    try:
        # Create environment
        env = OpenArmEnv(
            use_velocity=args.use_velocity, 
            action_scale=args.action_scale, 
            horizon=args.steps
        )
        print(f"[rl_train_joint] Environment created successfully. Action space: {env.action_space}, Observation space: {env.observation_space}")
        
        # Training loop
        returns = []
        for ep in range(args.episodes):
            obs, _ = env.reset()
            ep_ret = 0.0
            print(f"[rl_train_joint] Episode {ep+1}/{args.episodes} started")
            
            for t in range(args.steps):
                # Random action for now - you can replace this with your brain2rl agent
                action = env.action_space.sample()
                obs, r, term, trunc, info = env.step(action)
                ep_ret += r
                
                if t % 50 == 0:  # Print progress every 50 steps
                    print(f"  Step {t}: reward={r:.3f}, cumulative={ep_ret:.3f}")
                
                if term or trunc:
                    break
            
            returns.append(ep_ret)
            print(f"[rl_train_joint] Episode {ep+1}/{args.episodes} finished: return = {ep_ret:.3f}")
        
        # Print training summary
        print(f"\n[rl_train_joint] Training completed!")
        print(f"[rl_train_joint] Mean return: {np.mean(returns):.3f}")
        print(f"[rl_train_joint] Std return: {np.std(returns):.3f}")
        print(f"[rl_train_joint] Min return: {np.min(returns):.3f}")
        print(f"[rl_train_joint] Max return: {np.max(returns):.3f}")
        
    except Exception as e:
        print(f"[rl_train_joint] Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        try:
            env.close()
            print("[rl_train_joint] Environment closed successfully")
        except Exception as e:
            print(f"[rl_train_joint] Warning: Error closing environment: {e}")

if __name__ == "__main__":
    main()
