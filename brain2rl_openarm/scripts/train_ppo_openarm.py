#!/usr/bin/env python3
"""
Script to train PPO agents on the OpenArm environment.
This script integrates your brain2rl PPO implementation with the OpenArm environment.
"""

import argparse
import sys
import os
import numpy as np
import torch
import time

# Add the brain2rl directory to the Python path
brain2rl_path = os.path.expanduser('~/brain2rl')
if brain2rl_path not in sys.path:
    sys.path.insert(0, brain2rl_path)

# Add the specific paths for the PPO agent
models_path = os.path.join(brain2rl_path, 'models')
if models_path not in sys.path:
    sys.path.insert(0, models_path)

try:
    from brain2rl_openarm.envs.openarm_env import OpenArmEnv
    print("[train_ppo_openarm] Successfully imported OpenArmEnv")
except ImportError as e:
    print(f"Error importing OpenArmEnv: {e}")
    print("Make sure the brain2rl_openarm package is properly built and sourced")
    sys.exit(1)

def create_ppo_agent(env, device="cpu"):
    """Create a PPO agent from brain2rl"""
    try:
        print("[train_ppo_openarm] Creating PPO agent...")
        
        # Import the PPO agent
        from models.rl.agents.ppo import PPOAgent
        
        # Get environment dimensions
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        
        print(f"[train_ppo_openarm] Creating PPO agent with obs_dim={obs_dim}, act_dim={act_dim}")
        
        # Create PPO agent
        agent = PPOAgent(observation_dim=obs_dim, action_dim=act_dim, device=device)
        
        return agent
        
    except ImportError as e:
        print(f"[train_ppo_openarm] Error importing PPO agent: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[train_ppo_openarm] Error creating PPO agent: {e}")
        sys.exit(1)

def scale_action_to_env(action, env):
    """Scale action from PPO output [-1,1] to environment action space"""
    action_space_low = env.action_space.low
    action_space_high = env.action_space.high
    
    # Scale from [-1, 1] to [action_space_low, action_space_high]
    scaled_action = (action + 1) / 2 * (action_space_high - action_space_low) + action_space_low
    
    return scaled_action

def main():
    p = argparse.ArgumentParser(description='Train PPO agent on OpenArm environment')
    p.add_argument('--episodes', type=int, default=100, help='Number of training episodes')
    p.add_argument('--steps', type=int, default=300, help='Maximum steps per episode')
    p.add_argument('--action-scale', type=float, default=0.03, help='Action scaling factor')
    p.add_argument('--use-velocity', action='store_true', help='Include velocity in observations')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='Device to use for training')
    p.add_argument('--save-interval', type=int, default=20, help='Save model every N episodes')
    p.add_argument('--save-dir', type=str, default='~/openarm_models', help='Directory to save models')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    args = p.parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print(f"[train_ppo_openarm] Starting PPO training on OpenArm")
    print(f"[train_ppo_openarm] Episodes: {args.episodes}, Steps per episode: {args.steps}")
    print(f"[train_ppo_openarm] Action scale: {args.action_scale}, Use velocity: {args.use_velocity}")
    print(f"[train_ppo_openarm] Device: {args.device}")
    
    try:
        # Create environment
        env = OpenArmEnv(
            use_velocity=args.use_velocity,
            action_scale=args.action_scale,
            horizon=args.steps
        )
        print(f"[train_ppo_openarm] Environment created successfully")
        
        # Create PPO agent
        agent = create_ppo_agent(env, device=args.device)
        print(f"[train_ppo_openarm] PPO agent created successfully")
        
        # Create save directory
        save_dir = os.path.expanduser(args.save_dir)
        os.makedirs(save_dir, exist_ok=True)
        print(f"[train_ppo_openarm] Models will be saved to: {save_dir}")
        
        # Training loop
        returns = []
        best_return = float('-inf')
        
        for ep in range(args.episodes):
            obs, _ = env.reset()
            ep_ret = 0.0
            episode_transitions = []
            
            print(f"[train_ppo_openarm] Episode {ep+1}/{args.episodes} started")
            
            for t in range(args.steps):
                # Get action from PPO agent
                action, action_info = agent.get_action(obs, training=True)
                
                # Scale action to environment action space
                scaled_action = scale_action_to_env(action, env)
                
                # Take step in environment
                next_obs, r, term, trunc, info = env.step(scaled_action)
                ep_ret += r
                
                # Store transition for PPO update
                episode_transitions.append({
                    'obs': obs,
                    'action': action,  # Store original action for PPO
                    'reward': r,
                    'next_obs': next_obs,
                    'done': term or trunc,
                    'action_info': action_info
                })
                
                obs = next_obs
                
                if t % 50 == 0:
                    print(f"  Step {t}: reward={r:.3f}, cumulative={ep_ret:.3f}")
                
                if term or trunc:
                    break
            
            # Update PPO agent with episode data
            if len(episode_transitions) > 0:
                print(f"[train_ppo_openarm] Updating PPO agent with {len(episode_transitions)} transitions")
                
                # Store transitions in agent memory
                for transition in episode_transitions:
                    agent.store_transition(
                        transition['obs'],
                        transition['action'],
                        transition['reward'],
                        transition['next_obs'],
                        transition['done'],
                        transition['action_info']
                    )
                
                # Update the agent
                try:
                    agent.update()
                    print(f"[train_ppo_openarm] PPO update completed")
                except Exception as e:
                    print(f"[train_ppo_openarm] Warning: PPO update failed: {e}")
            
            returns.append(ep_ret)
            print(f"[train_ppo_openarm] Episode {ep+1}/{args.episodes} finished: return = {ep_ret:.3f}")
            
            # Save model periodically
            if (ep + 1) % args.save_interval == 0:
                model_path = os.path.join(save_dir, f'ppo_openarm_ep{ep+1}.pth')
                try:
                    torch.save({
                        'episode': ep + 1,
                        'policy_net': agent.policy_net.state_dict(),
                        'value_net': agent.value_net.state_dict(),
                        'returns': returns,
                        'best_return': best_return
                    }, model_path)
                    print(f"[train_ppo_openarm] Model saved to {model_path}")
                except Exception as e:
                    print(f"[train_ppo_openarm] Warning: Failed to save model: {e}")
            
            # Update best return
            if ep_ret > best_return:
                best_return = ep_ret
                # Save best model
                best_model_path = os.path.join(save_dir, 'ppo_openarm_best.pth')
                try:
                    torch.save({
                        'episode': ep + 1,
                        'policy_net': agent.policy_net.state_dict(),
                        'value_net': agent.value_net.state_dict(),
                        'returns': returns,
                        'best_return': best_return
                    }, best_model_path)
                    print(f"[train_ppo_openarm] New best model saved: {best_return:.3f}")
                except Exception as e:
                    print(f"[train_ppo_openarm] Warning: Failed to save best model: {e}")
        
        # Save final model
        final_model_path = os.path.join(save_dir, 'ppo_openarm_final.pth')
        try:
            torch.save({
                'episode': args.episodes,
                'policy_net': agent.policy_net.state_dict(),
                'value_net': agent.value_net.state_dict(),
                'returns': returns,
                'best_return': best_return
            }, final_model_path)
            print(f"[train_ppo_openarm] Final model saved to {final_model_path}")
        except Exception as e:
            print(f"[train_ppo_openarm] Warning: Failed to save final model: {e}")
        
        # Print training summary
        print(f"\n[train_ppo_openarm] Training completed!")
        print(f"[train_ppo_openarm] Mean return: {np.mean(returns):.3f}")
        print(f"[train_ppo_openarm] Std return: {np.std(returns):.3f}")
        print(f"[train_ppo_openarm] Min return: {np.min(returns):.3f}")
        print(f"[train_ppo_openarm] Max return: {np.max(returns):.3f}")
        print(f"[train_ppo_openarm] Best return: {best_return:.3f}")
        
    except Exception as e:
        print(f"[train_ppo_openarm] Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        try:
            env.close()
            print("[train_ppo_openarm] Environment closed successfully")
        except Exception as e:
            print(f"[train_ppo_openarm] Warning: Error closing environment: {e}")

if __name__ == "__main__":
    main()
