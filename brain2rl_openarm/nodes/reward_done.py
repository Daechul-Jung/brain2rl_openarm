import numpy as np

class RewardDone:
    def __init__(self, success_thresh = 0.03, max_step = 3000):
        self.success_thresh = float(success_thresh)
        self.max_step = int(max_step)

    def compute(self, t: int, ee_pos: np.ndarray, cup_pos: np.ndarray):
        dist = float(np.linalg.norm(ee_pos - cup_pos))

        reward = -dist
        success = dist < self.success_thresh
        done = success or (t >= self.max_step -1)
        info = {'dist': dist, 'success': success}
        return reward, done, info