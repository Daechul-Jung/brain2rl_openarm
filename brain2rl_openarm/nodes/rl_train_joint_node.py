import rclpy
from rclpy.node import Node
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from .joint_order_node import JointOrder
from .obs_builder import ObsBuilder

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