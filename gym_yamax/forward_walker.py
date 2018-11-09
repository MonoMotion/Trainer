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
        self.fail_ratio = 1 / 3
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
        cost = 0
        for a, j in zip(action, self.ordered_joints):
            target = j.current_position()[0] + a
            target_clipped = max(-math.pi / 2, min(target, math.pi / 2))
            cost += abs(target - target_clipped)
            # TODO: Calculate kp, kd, and maxForce correctly
            j.set_servo_target(target_clipped, 0.1, 1.0, 100000)
        return cost

    def calc_state(self):
        body_pose = self.robot_body.pose()
        parts_xyz = np.array( [p.pose().xyz() for p in self.parts.values()] ).flatten()
        self.body_xyz = (parts_xyz[0::3].mean(), parts_xyz[1::3].mean(), body_pose.xyz()[2])

        jointStates = [j.current_position()[0] for j in self.ordered_joints]
        euler = self.robot_body.pose().rpy()
        return jointStates + list(euler)

    def calc_energy_cost(self):
        return sum(abs(j.current_position()[1]) for j in self.ordered_joints)

    def calc_tilt_cost(self):
        def euler_to_axis(euler):
            c = [math.cos(a / 2) for a in euler]
            s = [math.sin(a / 2) for a in euler]
            axis_angle = 2 * math.acos(reduce(mul, c) - reduce(mul, s))
            return axis_angle

        body_tilt = euler_to_axis(self.robot_body.pose().rpy())
        foot_tilt_sum = sum(euler_to_axis(self.parts[name].pose().rpy()) for name in self.foot_list)
        return body_tilt + foot_tilt_sum

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

    def _step(self, action):
        out_of_range_cost = 0

        # if multiplayer, action first applied to all robots,
        # then global step() called, then _step() for all robots with the same actions
        if not self.scene.multiplayer:
            self.scene.cpp_world.step(1)
            out_of_range_cost = self.apply_action(action)
            self.scene.global_step()

        state = self.calc_state()
        x, y, z = self.get_position()

        fell_over = self.initial_z - z > self.fail_ratio * self.initial_z
        done = fell_over
        feetCollisionCost = self.calc_feet_collision_cost()
        energy_cost = self.calc_energy_cost()
        tilt_cost = self.calc_tilt_cost()

        rewards_dict = {
            'height_cost': 5 * min(z - self.initial_z, 0),
            'out_of_range_cost': - 0.1 * out_of_range_cost,
            'feet_collision_cost': 0.1 * feetCollisionCost,
            'energy_cost': - 0.01 * energy_cost,
            'tilt_cost': - 0.01 * tilt_cost,
            'progress': 10 * (x - self._last_x),
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
