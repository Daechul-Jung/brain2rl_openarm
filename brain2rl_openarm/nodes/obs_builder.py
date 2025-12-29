import numpy as np

class ObsBuilder:
    """
    Define how ROS messages become the obs my brain2rl policy expects.
    Default: obs = [q, dq]
    """
    def __init__(self, use_effort = False):
        self.use_effort = use_effort

    def build(self, q, dq, effort = None):
        parts = [q, dq]
        if self.use_effort:
            parts.append(effort if effort is not None else np.zeros_like(q))
        return np.concatenate(parts, axis = 0).astype(np.float32)   