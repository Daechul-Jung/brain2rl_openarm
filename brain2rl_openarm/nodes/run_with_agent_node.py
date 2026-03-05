# brain2rl_openarm/nodes/run_with_agent_node.py
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from brain2rl_openarm.nodes.joint_order import JointOrder
from brain2rl_openarm.nodes.obs_builder import ObsBuilder

class RunWithAgentNode(Node):
    """
    Runs a brain2rl agent (REPPO/VLA Thinker/etc.) online in Gazebo.
    - Build obs from ROS
    - agent.act(obs, maybe instruction) -> action
    - publish to controller
    """
    def __init__(self):
        super().__init__("run_with_agent_node")

        self.declare_parameter("agent_ckpt", "")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("command_topic", "/joint_group_position_controller/commands")
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("joint_names", ["joint1","joint2","joint3","joint4","joint5","joint6"])
        self.declare_parameter("max_delta_rad", 0.05)
        self.declare_parameter("instruction", "")  # for VLA-style

        self.ckpt = self.get_parameter("agent_ckpt").value
        self.js_topic = self.get_parameter("joint_states_topic").value
        self.cmd_topic = self.get_parameter("command_topic").value
        self.rate = float(self.get_parameter("control_rate_hz").value)
        self.joint_names = list(self.get_parameter("joint_names").value)
        self.max_delta = float(self.get_parameter("max_delta_rad").value)
        self.instruction = self.get_parameter("instruction").value

        self.jorder = JointOrder(self.joint_names)
        self.obs_builder = ObsBuilder(use_effort=False)

        self.q = None
        self.dq = None
        self.sub = self.create_subscription(JointState, self.js_topic, self.cb_js, 10)
        self.pub = self.create_publisher(Float64MultiArray, self.cmd_topic, 10)

        # --- Load your brain2rl agent here ---
        # Example (YOU EDIT):
        # from models.rl.thinkers.thinkingAgent import ThinkingAgent
        # self.agent = ThinkingAgent.load(self.ckpt)
        self.agent = None

        self.timer = self.create_timer(1.0 / self.rate, self.step)

    def cb_js(self, msg: JointState):
        q, dq, _ = self.jorder.reorder(msg)
        self.q, self.dq = q, dq

    def step(self):
        if self.q is None:
            return

        obs = self.obs_builder.build(self.q, self.dq)  # (12,)
        # --- Call agent to produce action ---
        # Replace with your actual agent API:
        # action = self.agent.act(obs, instruction=self.instruction)
        # For now:
        action = np.zeros(len(self.q), dtype=np.float32)

        delta = np.clip(action, -1, 1) * self.max_delta
        q_cmd = self.q + delta

        msg = Float64MultiArray()
        msg.data = q_cmd.tolist()
        self.pub.publish(msg)

def main():
    rclpy.init()
    node = RunWithAgentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()