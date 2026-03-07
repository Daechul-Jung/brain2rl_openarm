import rclpy
from rclpy.node import Node
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from brain2rl_openarm.nodes.ros_env_node import RosOpenArmEnv
from brain2rl_openarm.nodes.joint_order import JointOrder
from brain2rl_openarm.nodes.obs_builder import ObsBuilder

class SimplePolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )
    def forward(self, x):
        return torch.tanh(self.net(x))
    

class RLTrainJointNode(Node):
    """
    Minimal ROS training Node 
    """

    def __init__(self):
        super().__init__('rl_train_joint_node')

        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('command_topic', '/joint_group_position_controller/commands')
        self.declare_parameter('control_rate_hz', 20.0)
        self.declare_parameter('joint_names', ["joint1","joint2","joint3","joint4","joint5","joint6"])
        self.declare_parameter('episodes', 5000)
        self.declare_parameter('steps_per_episode', 5000)
        self.declare_parameter('max_delta_rad', 0.05)
        self.declare_parameter('q_target', [-0.2, 0.2, 0.15, 0.0, 0.0, 0.0])
        self.declare_parameter('save_path', 'ros_joint_policy.pt')

        js_topic = self.get_parameter("joint_states_topic").value
        cmd_topic = self.get_parameter("command_topic").value
        rate = float(self.get_parameter("control_rate_hz").value)
        joint_names = list(self.get_parameter("joint_names").value)
        self.episodes = int(self.get_parameter("episodes").value)
        self.steps_per_ep = int(self.get_parameter("steps_per_episode").value)
        self.max_delta = float(self.get_parameter("max_delta_rad").value)
        self.q_target = np.array(self.get_parameter("q_target").value, dtype=np.float32)
        self.save_path = self.get_parameter("save_path").value

        self.env = RosOpenArmEnv(
            node=self, 
            joint_names=joint_names,
            cmd_topic=cmd_topic,
            js_topic=js_topic,
            rate_hz=rate,
            use_effort=False
        )

        dof = len(joint_names)
        self.obs_dim = dof * 2
        self.act_dim = dof

        self.policy = SimplePolicy(self.obs_dim, self.act_dim)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=3e-4)

        self.started = False
        self.timer = self.create_timer(2.0, self.start_once)

    def start_once(self):
        if self.started:
            return
        self.started = True
        self.get_logger().info("Starting training...")
        self.train_loop()

    def reward(self, q):
        return -float(np.linalg.norm(q - self.q_target))
    
    def train_loop(self):
        self.policy.train()

        for ep in range(self.episodes):
            obs = self.env.reset_home()
            ep_return = 0.0
            losses = []

            for t in range(self.steps_per_ep):
                obs_t = torch.from_numpy(obs).float().unsqueeze(0)
                a = self.policy(obs_t).squeeze(0)  # [-1,1]
                a = (a + 0.1 * torch.randn_like(a)).clamp(-1, 1)

                # delta position control
                delta = a.detach().cpu().numpy() * self.max_delta
                q_cmd = self.env.q + delta

                obs = self.env.step_position(q_cmd)
                r = self.reward(self.env.q)
                ep_return += r

                # toy loss (replace with PPO/REPPO)
                losses.append(torch.tensor((np.linalg.norm(self.env.q - self.q_target))**2, dtype=torch.float32))

            loss = torch.stack(losses).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            self.get_logger().info(f"EP {ep}: return={ep_return:.3f} loss={loss.item():.6f}")

        torch.save({"state_dict": self.policy.state_dict(),
                    "obs_dim": self.obs_dim,
                    "act_dim": self.act_dim}, self.save_path)
        self.get_logger().info(f"Saved policy -> {self.save_path}")


def main():
    rclpy.init()
    node = RLTrainJointNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()