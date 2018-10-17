from roboschool.scene_stadium import SinglePlayerStadiumScene
from roboschool.multiplayer import SharedMemoryClientEnv
from baselines import logger
import numpy as np
import math
from functools import reduce
from operator import mul


class ForwardWalker(SharedMemoryClientEnv):
    def __init__(self, servo_angular_speed=0.14):
        self._angular_velocity_limit = math.pi / (servo_angular_speed * 3)
        self.fail_threshold = 45 * math.pi / 180
        self.success_x_threshold = 3
        self.start_pos_x, self.start_pos_y, self.start_pos_z = 0, 0, 0
        self.camera_x = 0

    def create_single_player_scene(self):
        return SinglePlayerStadiumScene(gravity=9.8,
                                        timestep=0.0165/4,
                                        frame_skip=3)
    random_initial_joints = False

    def robot_specific_reset(self):
        for j in self.ordered_joints:
            j.reset_current_position(self.np_random.uniform(
                low=-0.1, high=0.1) if self.random_initial_joints else 0, 0)
        self.feet = [self.parts[f] for f in self.foot_list]
        self.feet_contact = np.array(
            [0.0 for f in self.foot_list], dtype=np.float32)
        self.scene.actor_introduce(self)
        x, _, _ = self.get_position()
        self._last_x = x
        self.current_ts = 0
        # self.initial_z = None

    def move_robot(self, init_x, init_y, init_z):
        """
        Used by multiplayer stadium to move sideways, to another running lane.
        """
        self.cpp_robot.query_position()
        pose = self.cpp_robot.root_part.pose()
        # Works because robot loads around (0,0,0),
        # and some robots have z != 0 that is left intact
        pose.move_xyz(init_x, init_y, init_z)
        self.cpp_robot.set_pose(pose)
        self.start_pos_x, self.start_pos_y, self.start_pos_z = init_x, init_y, init_z

    def apply_action(self, action):
        assert(np.isfinite(action).all())
        for a, j in zip(action, self.ordered_joints):
            # TODO: Calculate kp, kd, and maxForce correctly
            j.set_servo_target(j.current_position()[0] + a, 0.1, 1.0, 100000)

    def calc_state(self):
        body_pose = self.robot_body.pose()
        parts_xyz = np.array( [p.pose().xyz() for p in self.parts.values()] ).flatten()
        self.body_xyz = (parts_xyz[0::3].mean(), parts_xyz[1::3].mean(), body_pose.xyz()[2])

        jointStates = [j.current_position()[0] for j in self.ordered_joints]
        euler = self.parts[self.hip_part].pose().rpy()
        return jointStates + list(euler)

    def get_position(self):
        self.cpp_robot.query_position()
        pose = self.cpp_robot.root_part.pose()
        return pose.xyz()

    foot_collision_cost = -1
    foot_ground_object_names = set(["floor"])

    def calc_feet_collision_cost(self):
        feet_collision_cost = 0.0
        for i, f in enumerate(self.feet):
            contact_names = set(x.name for x in f.contact_list())
            self.feet_contact[i] = 1.0 if (
                self.foot_ground_object_names & contact_names) else 0.0
            if contact_names - self.foot_ground_object_names:
                feet_collision_cost += self.foot_collision_cost
        return feet_collision_cost

    def get_legs_orientation(self):
        def get_roll(name):
            euler = self.parts[name].pose().rpy()
            return euler[0]  # roll
        return (get_roll(self.left_leg), get_roll(self.right_leg))

    def _step(self, action):
        # if multiplayer, action first applied to all robots,
        # then global step() called, then _step() for all robots with the same actions
        if not self.scene.multiplayer:
            self.scene.cpp_world.step(1)
            self.apply_action(action)
            self.scene.global_step()

        state = self.calc_state()
        x, y, z = self.get_position()

        euler = self.cpp_robot.root_part.pose().rpy()
        c = [math.cos(a / 2) for a in euler]
        s = [math.sin(a / 2) for a in euler]
        axisAngle = 2 * math.acos(reduce(mul, c) - reduce(mul, s))
        fell_over = axisAngle > self.fail_threshold
        done = fell_over or self.current_ts > 100
        feetCollisionCost = self.calc_feet_collision_cost()

        rewards_dict = {
            'zdiff': 10 * (z - self.initial_z),
            'feet_collision_cost': 0.1 * feetCollisionCost,
            'progress': - 10 * (self._last_x - x),
        }

        self.rewards = list(rewards_dict.values())

        state = np.array(state)

        # Log reward values
        for k, v in rewards_dict.items():
            logger.logkv_mean(k + '_mean', v)
            if done:
                logger.logkv_mean('last_' + k + '_mean', v)

        logger.logkv_mean('xpos_mean', x)
        if done:
            logger.logkv_mean('last_xpos_mean', x)

        self.current_ts += 1
        reward_sum = sum(self.rewards) if not fell_over else -1

        # for RoboschoolUrdfEnv
        self.frame += 1
        self.done += done
        self.reward += reward_sum
        self.HUD(state, action, done)

        self._last_x = x
        return state, reward_sum, done, {}

    def camera_adjust(self):
        self.camera_simple_follow()

    def camera_simple_follow(self):
        x, y, z = self.body_xyz
        self.camera_x = 0.98*self.camera_x + (1-0.98)*x
        self.camera.move_and_look_at(self.camera_x, y-2.0, 1.4, x, y, 1.0)
