import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

from joint_order import JointOrder
from obs_builder import ObsBuilder

class RosOpenArmEnv:
    """
    Python env wrapper using ROS topics (used inside a trainer node)
    """
    def __init__(self, node: Node, joint_names, cmd_topic, js_topic, rate_hz = 20.0, use_effort = False):
        self.node = Node
        self.rate_hz = rate_hz
        self.pub = node.create_publisher(msg_type=Float64MultiArray, topic=cmd_topic, qos_profile=10)
        self.jorder = JointOrder(joint_names)
        self.obs_builder = ObsBuilder(use_effort)

        self.q = None
        self.dq = None
        self.eff = None 
        ## _cb_js is fallback function 
        self.sub = node.create_subscription(JointState, js_topic, self._cb_js, 10)

    def _cb_js(self, msg: JointState):
        q, dq, eff = self.jorder.reorder(msg = msg)

        self.q, self.dq, self.eff = q, dq, eff


    def wait_ready(self):
        while rclpy.ok() and self.q is None:
            rclpy.spin_once(self.node, timeout_sec=0.1)

    def reset_home(self):
        cmd = Float64MultiArray()
        cmd.data = np.zeros_like(self.q).tolist()
        self.pub.publish(cmd)
        rclpy.spin_once(self.node, timeout_sec=0.5)
        return self.obs_builder.build(self.q, self.dq, self.eff)
    

    def step_posittion(self, q_cmd: np.ndarray):
        cmd = Float64MultiArray()
        cmd.data = q_cmd.astype(np.float32).tolist()
        self.pub.publish(cmd)
        rclpy.spin_once(self.node, timeout_sec=1.0 / self.rate_hz)
        return self.obs_builder.build(self.q, self.dq, self.eff)