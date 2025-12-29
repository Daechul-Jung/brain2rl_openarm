import numpy as np
from sensor_msgs.msg import JointState

class JointOrder:
    def __init__(self, desired_joint_names):
        self.desired = list(desired_joint_names)
        self.index_map = None ## filled on first message

    def _build_map(self, msg: JointState):
        name_to_idx = {n : i for i, n in enumerate(msg.name)}
        self.index_map = [name_to_idx[n] for n in self.desired]

    def reorder(self, msg: JointState):
        if self.index_map is None:
            self._build_map(msg)

        pos = np.array(msg.position, dtype=np.float32)[self.index_map]

        if msg.velocity: 
            vel = np.array(msg.velocity, dtype = np.float32)[self.index_map]
        else:
            vel = np.zeros_like(pos, dtype=np.float32)

        if msg.effort:
            effort = np.array(msg.effort, dtype=np.float32)[self.index_map]
        else:
            effort = np.zeros_like(pos, dtype = np.float32)

        return pos, vel, effort
    
    