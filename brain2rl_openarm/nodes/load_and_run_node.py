"""
load_and_run_node.py
Loads a REPPO / PPO checkpoint trained in MuJoCo or ManiSkill and runs
inference on the Gazebo OpenArm.

Handles obs-dim mismatch by zero-padding the ROS observation to match the
checkpoint's expected obs_dim (MuJoCo/ManiSkill observations are larger).

Launch:
    ros2 launch brain2rl_openarm openarm_rl_gazebo.launch.py \
        load_run:=true ckpt_path:=/path/to/checkpoint.pth
"""
import os
import sys
import threading

sys.path.insert(0, '/home/jdcjdb/brain2rl')

import numpy as np
import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from brain2rl_openarm.nodes.joint_order import JointOrder
from brain2rl_openarm.nodes.obs_builder import ObsBuilder


class LoadAndRunNode(Node):
    """
    Inference node: loads a brain2rl REPPO checkpoint and publishes actions.

    Obs-dim adaptation
    ------------------
    MuJoCo/ManiSkill checkpoints were trained with larger obs vectors.
    We zero-pad the 12-dim ROS obs (q + qd) to the checkpoint's obs_dim.
    You can also override pad_dim with the checkpoint's actual obs_dim.
    """

    def __init__(self):
        super().__init__('load_and_run_node')

        self.declare_parameter('ckpt_path',          '')
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('command_topic',
                               '/joint_group_position_controller/commands')
        self.declare_parameter('joint_names',
                               ['joint1','joint2','joint3',
                                'joint4','joint5','joint6'])
        self.declare_parameter('max_delta_rad', 0.05)
        self.declare_parameter('control_rate_hz', 20.0)
        self.declare_parameter('pad_dim', -1)   # -1 = auto-detect from checkpoint

        ckpt_path   = self.get_parameter('ckpt_path').value
        js_topic    = self.get_parameter('joint_states_topic').value
        cmd_topic   = self.get_parameter('command_topic').value
        joint_names = list(self.get_parameter('joint_names').value)
        self.max_delta  = float(self.get_parameter('max_delta_rad').value)
        rate_hz         = float(self.get_parameter('control_rate_hz').value)
        self.pad_dim    = int(self.get_parameter('pad_dim').value)

        self.jorder      = JointOrder(joint_names)
        self.obs_builder = ObsBuilder(use_effort=False)
        self.q           = None
        self.dq          = None

        self.sub = self.create_subscription(JointState, js_topic, self._cb_js, 10)
        self.pub = self.create_publisher(Float64MultiArray, cmd_topic, 10)

        self.agent   = None
        self.obs_dim = None

        if ckpt_path:
            self._load_agent(ckpt_path)
        else:
            self.get_logger().warn('[load_run] No ckpt_path set -- publishing zeros.')

        self.create_timer(1.0 / rate_hz, self._step)

    # ------------------------------------------------------------------ #
    def _load_agent(self, path: str):
        from models.rl.agents.reppo import RePPOAgent, EmpiricalNormalizer
        try:
            ckpt = torch.load(path, map_location='cpu')
            # Infer obs/action dim from actor weights
            actor_w = ckpt.get('actor', ckpt)
            first_key = next(k for k in actor_w if 'weight' in k)
            act_in  = actor_w[first_key].shape[1]   # obs_dim
            # Find action dim from last linear layer of actor
            last_key = [k for k in actor_w if 'weight' in k][-1]
            act_out  = actor_w[last_key].shape[0]   # could be action*2 (mean+std)

            # Use pad_dim if explicitly set, else use checkpoint obs_dim
            self.obs_dim = act_in if self.pad_dim < 0 else self.pad_dim

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            obs_norm  = EmpiricalNormalizer(shape=self.obs_dim, device=device)
            cobs_norm = EmpiricalNormalizer(shape=self.obs_dim, device=device)
            agent = RePPOAgent(
                observation_dim=self.obs_dim,
                action_dim=6,
                obs_normalizer=obs_norm,
                critic_obs_normalizer=cobs_norm,
                device=device,
            )
            agent.load(path, strict=False, load_optim=False)
            self.agent  = agent
            self.device = device
            self.get_logger().info(
                f'[load_run] Loaded checkpoint {path}  '
                f'obs_dim={self.obs_dim}  action_dim=6')
        except Exception as e:
            self.get_logger().error(f'[load_run] Failed to load: {e}')

    # ------------------------------------------------------------------ #
    def _cb_js(self, msg: JointState):
        q, dq, _ = self.jorder.reorder(msg)
        self.q, self.dq = q, dq

    # ------------------------------------------------------------------ #
    def _step(self):
        if self.q is None:
            return

        ros_obs = self.obs_builder.build(self.q, self.dq)  # (12,)

        if self.agent is None or self.obs_dim is None:
            action = np.zeros(len(self.q), dtype=np.float32)
        else:
            # Pad or truncate obs to match checkpoint obs_dim
            if len(ros_obs) < self.obs_dim:
                padded = np.zeros(self.obs_dim, dtype=np.float32)
                padded[:len(ros_obs)] = ros_obs
            else:
                padded = ros_obs[:self.obs_dim]

            action, _ = self.agent.get_action(padded, training=False)
            # action is in [-1, 1]; scale to delta radians
            action = action[:len(self.q)]

        delta = np.clip(action, -1.0, 1.0) * self.max_delta
        q_cmd = self.q + delta

        out = Float64MultiArray()
        out.data = q_cmd.tolist()
        self.pub.publish(out)


def main():
    rclpy.init()
    node = LoadAndRunNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
