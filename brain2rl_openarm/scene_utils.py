import time
from geometry_msgs.msg import PoseStamped

class Scene:
    def __init__(self, frame=None):
        self.robot = RobotCommander()
        self.scene = PlanningSceneInterface(synchronous=True)
        self.frame = frame or self.robot.get_planning_frame()

    def add_box(self, name='cup', size=(0.05,0.05,0.1), xyz=(0.5,0.0,0.2)):
        pose = PoseStamped()
        pose.header.frame_id = self.frame
        pose.pose.orientation.w = 1.0
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = xyz
        self.scene.add_box(name, pose, size)
        time.sleep(0.2)

    def remove(self, name):
        self.scene.remove_world_object(name)
        time.sleep(0.1)
