from forward_walker import ForwardWalker
from roboschool.gym_urdf_robot_env import RoboschoolUrdfEnv
from roboschool.scene_abstract import cpp_household
from roboschool.scene_stadium import SinglePlayerStadiumScene
import numpy as np


class RoboschoolYamaXForwardWalk(ForwardWalker, RoboschoolUrdfEnv):
    random_yaw = False
    foot_list = ["foot_right", "foot_left"]
    right_leg = "leg_right_2"
    left_leg = "leg_left_2"
    hip_part = "hip"

    def __init__(self):
        YamaXForwardWalker.__init__(self)
        RoboschoolUrdfEnv.__init__(self,
                                   "robot_models/yamax.urdf",
                                   "YamaX",
                                   action_dim=20, obs_dim=26,
                                   fixed_base=False,
                                   self_collision=True)

    def create_single_player_scene(self):
        # 8 instead of 4 here
        return SinglePlayerStadiumScene(gravity=9.8, timestep=0.0165/8, frame_skip=8)

    def robot_specific_reset(self):
        YamaXForwardWalker.robot_specific_reset(self)
        self.set_initial_orientation(yaw_center=0, yaw_random_spread=np.pi)
        self.head = self.parts["head"]

    random_yaw = False

    def set_initial_orientation(self, yaw_center, yaw_random_spread):
        cpose = cpp_household.Pose()
        if not self.random_yaw:
            yaw = yaw_center
        else:
            yaw = yaw_center + \
                self.np_random.uniform(
                    low=-yaw_random_spread, high=yaw_random_spread)

        cpose.set_xyz(self.start_pos_x, self.start_pos_y,
                      self.start_pos_z + 1.0)
        # just face random direction, but stay straight otherwise
        cpose.set_rpy(0, 0, yaw)
        self.cpp_robot.set_pose_and_speed(cpose, 0, 0, 0)
        self.initial_z = 1.5
