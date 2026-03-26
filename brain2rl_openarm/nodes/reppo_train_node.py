"""
reppo_train_node.py
Trains a RePPO agent on the Gazebo OpenArm sim via the OpenArmEnv gym wrapper.
"""
import os
import sys
import threading

# brain2rl path
sys.path.insert(0, '/home/jdcjdb/brain2rl')

import numpy as np
import torch
import rclpy
from rclpy.node import Node

from brain2rl_openarm.envs.openarm_env import OpenArmEnv
from models.rl.agents.reppo import RePPOAgent, EmpiricalNormalizer
from models.rl.utils.train import train_reppo


class RePPOTrainNode(Node):
    """
    ROS2 node that trains a RePPO agent on the Gazebo OpenArm.

    Run with:
        ros2 launch brain2rl_openarm openarm_rl_gazebo.launch.py reppo_train:=true
    """

    def __init__(self):
        super().__init__('reppo_train_node')

        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('command_topic',
                               '/joint_group_position_controller/commands')
        self.declare_parameter('joint_names',
                               ['joint1', 'joint2', 'joint3',
                                'joint4', 'joint5', 'joint6'])
        self.declare_parameter('total_steps', 200_000)
        self.declare_parameter('num_step',    128)
        self.declare_parameter('save_path',   'output/reppo_ros/checkpoint.pth')
        self.declare_parameter('action_scale', 0.05)
        self.declare_parameter('horizon',      200)

        self.total_steps  = self.get_parameter('total_steps').value
        self.num_step     = self.get_parameter('num_step').value
        self.save_path    = self.get_parameter('save_path').value
        self.action_scale = float(self.get_parameter('action_scale').value)
        self.horizon      = int(self.get_parameter('horizon').value)

        # Delayed start so Gazebo and controllers are ready
        self._started = False
        self.create_timer(12.0, self._start_once)

    # ------------------------------------------------------------------ #
    def _start_once(self):
        if self._started:
            return
        self._started = True
        self.get_logger().info('[reppo_train] Starting REPPO training thread ...')
        t = threading.Thread(target=self._train, daemon=True)
        t.start()

    # ------------------------------------------------------------------ #
    def _train(self):
        try:
            self._run_training()
        except Exception as e:
            self.get_logger().error(f'[reppo_train] Training failed: {e}')
            import traceback
            traceback.print_exc()

    # ------------------------------------------------------------------ #
    def _run_training(self):
        # --- Environment ---
        env = OpenArmEnv(
            use_velocity=True,
            action_scale=self.action_scale,
            horizon=self.horizon,
        )
        obs_dim    = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        device     = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.get_logger().info(
            f'[reppo_train] obs_dim={obs_dim}  action_dim={action_dim}  device={device}')

        obs_norm  = EmpiricalNormalizer(shape=obs_dim, device=device)
        cobs_norm = EmpiricalNormalizer(shape=obs_dim, device=device)

        agent = RePPOAgent(
            observation_dim=obs_dim,
            action_dim=action_dim,
            num_atoms=51,       # smaller range for ROS joint-space rewards
            vmin=-20.0,
            vmax=5.0,
            device=device,
            lr=3e-4,
            gamma=0.99,
            lmbda=0.95,
            kl_start=0.01,
            entropy_start=0.01,
            obs_normalizer=obs_norm,
            critic_obs_normalizer=cobs_norm,
        )

        self.get_logger().info('[reppo_train] Beginning train_reppo ...')
        returns = train_reppo(
            env=env,
            agent=agent,
            total_steps=self.total_steps,
            num_step=self.num_step,
            num_epoch=8,
            num_mini_batch=4,
        )

        os.makedirs(os.path.dirname(self.save_path) or '.', exist_ok=True)
        agent.save(self.save_path, extra={'returns': returns})
        self.get_logger().info(f'[reppo_train] Saved -> {self.save_path}')
        if returns:
            self.get_logger().info(
                f'[reppo_train] Mean return (last 20): {np.mean(returns[-20:]):.3f}')
        env.close()


def main():
    rclpy.init()
    node = RePPOTrainNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
