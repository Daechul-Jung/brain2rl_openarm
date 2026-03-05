import rclpy
import numpy as np
from rclpy.node import Node

import time, os, json

from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Float64MultiArray


class TrajRecorderNode(Node):
    def __init__(self):
        super().__init__('traj_recorder_node')

        self.declare_parameter("out_dir", "ros_trajs")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("command_topic", "/joint_group_position_controller/commands")
        self.declare_parameter("image_topic", "")
        self.declare_parameter("episode_id", 0)

        self.out_dir = self.get_parameter("out_dir").value
        self.js_topic = self.get_parameter("joint_states_topic").value
        self.cmd_topic = self.get_parameter("command_topic").value
        self.img_topic = self.get_parameter("image_topic").value
        self.episode_id = int(self.get_parameter("episode_id").value)

        os.makedirs(self.out_dir, exist_ok=True)

        self.t0 = True
        self.t = []
        self.joint_state = []
        self.commands = []
        self.images = []

        self.sub_js = self.create_subscription(JointState, self.js_topic, self.cb_js, 50)
        self.sub_cmd = self.create_subscription(Float64MultiArray, self.cmd_topic, self.cb_cmd, 50)
        self.sub_img = None

        if self.img_topic:
            self.sub_img = self.create_subscription(Image, self.img_topic, self.cb_img, 10)

        self.get_logger().info(f"Recording: out_dir={self.out_dir}, episode_id={self.episode_id}")

    def cb_js(self, msg: JointState):
        self.t.append(time.time() - self.t0)
        self.joint_state.append({
            "name": list(msg.name),
            "position": list(msg.position),
            "velocity": list(msg.velocity) if msg.velocity else [],
            "effort": list(msg.effort) if msg.effort else [],
        })

    def cb_cmd(self, msg: Float64MultiArray):
        self.commands.append(list(msg.data))

    def cb_img(self, msg: Image):
        # WARNING: images can be huge; keep disabled unless needed
        self.images.append({
            "height": msg.height,
            "width": msg.width,
            "encoding": msg.encoding,
            "data": bytes(msg.data),
        })

    def save(self):
        path = os.path.join(self.out_dir, f"ep_{self.episode_id:05d}.npz")
        meta = os.path.join(self.out_dir, f"ep_{self.episode_id:05d}.json")
        np.savez_compressed(
            path,
            t=np.array(self.t, dtype=np.float32),
            joint_states=np.array(self.joint_states, dtype=object),
            commands=np.array(self.commands, dtype=object),
        )
        with open(meta, "w") as f:
            json.dump({"joint_states_topic": self.js_topic, "command_topic": self.cmd_topic}, f)
        self.get_logger().info(f"Saved: {path}")

    def destroy_node(self):
        # auto-save on shutdown
        self.save()
        super().destroy_node()


def main():
    rclpy.init()
    node = TrajRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
