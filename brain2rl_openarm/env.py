#####################################
#### Gym environment for OpenArm ####
#####################################
import time
import numpy as np
import rclpy
from rclpy.node import Node
from moveit_commander import MoveGroupCommander, RobotCommander
from geometry_msgs.msg import PoseStamped
from scene_utils import Scene

class OpenArmReachEnv(Node):
    def __init__(self, group='left_arm', step_dt=0.3):
        super().__init__('openarm_env')
        self.robot = RobotCommander()
        self.scene = Scene()
        self.group = MoveGroupCommander(group)
        self.frame = self.robot.get_planning_frame()
        self.target = 'target'
        self.step_dt = step_dt

    def _spawn_target(self, xyz):
        pose = PoseStamped()
        pose.header.frame_id = self.frame
        pose.pose.orientation.w = 1.0
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = xyz
        self.scene.scene.remove_world_object(self.target)
        time.sleep(0.05)
        self.scene.scene.add_sphere(self.target, pose, radius=0.03)
        time.sleep(0.2)

    def reset(self):
        if 'home' in self.group.get_named_targets():
            self.group.set_named_target('home')
            self.group.go(wait=True)
        xyz = np.array([0.45 + 0.1*np.random.rand(),
                        0.10*(np.random.rand()-0.5),
                        0.18 + 0.1*np.random.rand()])
        self._spawn_target(tuple(xyz))
        return self._obs()

    def _obs(self):
        ee = self.group.get_current_pose().pose
        pose = self.scene.scene.get_object_poses([self.target])[self.target]
        obj = pose.pose
        return np.array([ee.position.x, ee.position.y, ee.position.z,
                         obj.position.x, obj.position.y, obj.position.z],
                        dtype=np.float32)

    def step(self, action):
        dx, dy, dz = np.clip(action, -0.03, 0.03)
        tgt = self.group.get_current_pose().pose
        tgt.position.x += float(dx); tgt.position.y += float(dy); tgt.position.z += float(dz)
        self.group.set_pose_target(tgt)
        plan = self.group.plan()
        self.group.execute(plan[1], wait=True)
        self.group.stop(); self.group.clear_pose_targets()
        obs = self._obs()
        ee, obj = obs[:3], obs[3:]
        dist = float(np.linalg.norm(ee - obj))
        reward = -dist
        done = dist < 0.02
        info = {'dist': dist}
        time.sleep(self.step_dt)
        return obs, reward, done, info
