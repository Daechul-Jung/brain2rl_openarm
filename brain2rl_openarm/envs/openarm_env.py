# ros2_ws/src/brain2rl_openarm/brain2rl_openarm/envs/openarm_env.py
import os
import threading
import time
from typing import Optional, Tuple, Dict

import gymnasium as gym
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


class _OpenArmNode(Node):
    def __init__(self, joint_state_topic='/joint_states',
                 cmd_topic='/joint_group_position_controller/commands'):
        super().__init__('openarm_env_node')
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._last_js: Optional[JointState] = None
        self._js_sub = self.create_subscription(
            JointState, joint_state_topic, self._on_js, qos)
        self._cmd_pub = self.create_publisher(Float64MultiArray, cmd_topic, 10)

    def _on_js(self, msg: JointState):
        self._last_js = msg

    def get_joint_state(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, list]]:
        js = self._last_js
        if js is None:
            return None
        pos = np.array(js.position, dtype=np.float32) if js.position else None
        vel = np.array(js.velocity, dtype=np.float32) if js.velocity else None
        eff = np.array(js.effort, dtype=np.float32) if js.effort else None
        return pos, vel, eff, list(js.name)

    def publish_positions(self, q: np.ndarray):
        msg = Float64MultiArray()
        msg.data = [float(x) for x in q]
        self._cmd_pub.publish(msg)


class OpenArmEnv(gym.Env):
    """
    Gym-style env for joint-space control with delta actions.
    Observation: concat(q, qd) or just q (configurable).
    Action: delta-q (radians) bounded by action_scale.
    Reward: -||q - q_goal||_2
    Done: horizon or close-to-goal.
    """
    metadata = {"render.modes": []}

    def __init__(self,
                 use_velocity=False,
                 action_scale=0.05,
                 horizon=200,
                 goal_q: Optional[np.ndarray] = None):
        super().__init__()
        self.use_velocity = use_velocity
        self.action_scale = float(action_scale)
        self.horizon = int(horizon)

        # --- ROS bringup (background spin) ---
        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_context = True
        else:
            self._owns_context = False

        self.node = _OpenArmNode()
        self._spin_thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self._spin_thread.start()

        # wait for first joint state
        self._joint_names = []
        self._q = None
        self._qd = None
        for _ in range(200):  # ~10s
            js = self.node.get_joint_state()
            if js is not None and js[0] is not None:
                self._q, self._qd, _, self._joint_names = js
                break
            time.sleep(0.05)
        if self._q is None:
            raise RuntimeError("OpenArmEnv: never received /joint_states")

        self.n_j = len(self._q)
        obs_dim = self.n_j * (2 if self.use_velocity else 1)

        self.action_space = gym.spaces.Box(
            low=-self.action_scale, high=self.action_scale, shape=(self.n_j,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self._t = 0
        self._home = self._q.copy()
        self._goal_q = goal_q.astype(np.float32) if goal_q is not None else self._home.copy()

    def _obs(self) -> np.ndarray:
        js = self.node.get_joint_state()
        if js is not None and js[0] is not None:
            self._q, self._qd, _, _ = js
        if self.use_velocity:
            return np.concatenate([self._q, self._qd]).astype(np.float32)
        return self._q.astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self._t = 0
        # simple reset: command home
        self.node.publish_positions(self._home)
        time.sleep(0.1)
        return self._obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # joint-space delta
        target_q = self._q + action
        self.node.publish_positions(target_q)

        # small settle time (tune for your controller update rate)
        time.sleep(0.02)

        obs = self._obs()
        # reward: get close to goal_q
        err = self._q - self._goal_q
        reward = -float(np.linalg.norm(err, ord=2))
        self._t += 1
        terminated = bool(np.linalg.norm(err) < 0.02)
        truncated = bool(self._t >= self.horizon)
        info = {"joint_names": self._joint_names}
        return obs, reward, terminated, truncated, info

    def render(self):
        return None

    def close(self):
        try:
            self.node.destroy_node()
        except Exception:
            pass
        if self._owns_context and rclpy.ok():
            rclpy.shutdown()
        if self._spin_thread.is_alive():
            time.sleep(0.1)
