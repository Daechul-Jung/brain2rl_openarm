import rclpy
from rclpy.node import Node
import numpy as np
import torch
import torch.nn as nn
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from .joint_order_node import JointOrder
from .obs_builder import ObsBuilder

def load_brain2rl_policy(policy_path: str):
    """
    Adapter loader:
    - If I have a brain2rl agent class with .load(), use it here 
    - Otherwise, load a TorchScript/StateDict model
    """

    ckpt = torch.load(policy_path, map_location='cuda')

    if isinstance(ckpt, torch.jit.RecursiveScriptModule):
        model = ckpt
        model.eval()
        return model, int(getattr(ckpt, 'obs_dim', 0)), int(getattr(ckpt, 'act_dim', 0))
    
    if 'state_dict' in ckpt and 'obs_dim' in ckpt and 'act_dim' in ckpt:
        obs_dim = int(ckpt['obs_dim'])
        act_dim = int(ckpt['act_dim'])
        
        class MLP(torch.nn.Module):
            def __init__(self, obs_dim, act_dim, hidden = 256):
                super().__init__()

                self.net = nn.Sequential(
                    nn.Linear(obs_dim, hidden), nn.ReLU(),
                    nn.Linear(hidden, hidden), nn.ReLU(),
                    nn.Linear(hidden, act_dim)
                )
            def forward(self, observation):
                return torch.tanh(self.net(observation))
            
        model = MLP(obs_dim, act_dim, 256)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        return model, obs_dim, act_dim


class PolicyRunnerNode(Node):
    def __init__(self):
        super().__init__('policy_runner_node')

        ## parameters
        self.declare_parameter('policy_path', 'policy.pt')
        self.declare_parameter('joint_state_topic', '/joint_states')
        self.declare_parameter('command_topic', '/joint_group_position_controller/commands')
        self.declare_parameter('control_rate_hz', 20.0)
        self.declare_parameter('joint_names', ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'])
        self.declare_parameter('action_mode', 'delta_position')
        self.declare_parameter('max_delta_rad', 0.05)
        self.declare_parameter('use_effort', False) 

        self.policy_path = self.get_parameter('policy_path').value
        self.js_topic = self.get_parameter('joint_state_topic').value 
        self.cmd_topic = self.get_parameter('command_topic').value
        self.rate = float(self.get_parameter('control_rate_hz').value)
        self.joint_names = list(self.get_parameter('joint_namse').value)
        self.action_mode = self.get_parameter('action_mode').value
        self.max_delta = float(self.get_parameter('max_delta_rad').value)
        self.use_effort = bool(self.get_parameter('use_effort').value)

        self.model, self.obs_dim, self.act_dim = load_brain2rl_policy(self.policy_path)
        self.get_logger().info(f'Loaded Policy: {self.policy_path} (obs_dim = {self.obs_dim}, act_dim = {self.act_dim})')

        self.joint_order = JointOrder(self.joint_names)
        self.obs_builder = ObsBuilder(self.use_effort)

        self.q = None
        self.dq = None 
        self.effort = None 

        self.sub = self.create_subscription(JointState, self.js_topic, self.callback_js, 10)
        self.pub = self.create_publisher(Float64MultiArray, self.cmd_topic, 10)
        self.timer = self.create_timer(1.0 / self.rate, self.step)


    def callback_js(self, msg: JointState):
        q, dq, effort = self.joint_order(msg)

        self.q, self.dq, self.effort  = q, dq, effort

    def step(self):
        if self.q is None:
            return 
        
        obs = self.obs_builder.build(self.q, self.dq, self.effort)
        if self.obs_dim and obs.shape[0] != self.obs_dim: ## since it is 1D array 
            if obs.shape[0] > self.obs_dim:
                obs = obs[:self.obs_dim]
            else:
                obs = np.pad(obs, (0, self.obs_dim - obs.shape[0]))

        obs_t = torch.from_numpy(obs).float().unsqueeze(0) ## [1, obs_dim]
        with torch.no_grad():
            action = self.model(obs_t).squeeze(0).cpu().numpy() ### [act_dim, ] \in [-1, 1]

        if self.action_mode == 'delta_position':
            delta = np.clip(action, -1.0, 1.0) * self.max_delta
            q_cmd = self.q[:len(delta)] + delta
        else:
            q_cmd = action

        msg = Float64MultiArray()
        msg.data = q_cmd.tolist()
        self.pub.publish(msg)

def main():
    rclpy.init()
    node = PolicyRunnerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

