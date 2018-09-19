from roboschool.scene_stadium import SinglePlayerStadiumScene
from roboschool.multiplayer import SharedMemoryClientEnv
import gym
import gym.spaces
import numpy as np
import math
from functools import reduce
from operator import mul


class YamaXForwardWalker(SharedMemoryClientEnv):
    def __init__(self, servo_angular_speed=0.14):
        self._updateDelay = frame_delay

        self._angular_velocity_limit = math.pi / (servo_angular_speed * 3)
        self.fail_threshold = 45 * math.pi / 180
        self.success_x_threshold = 3

    def create_single_player_scene(self):
        return SinglePlayerStadiumScene(gravity=9.8, timestep=0.0165/4, frame_skip=4)

    def robot_specific_reset(self):
        for j in self.ordered_joints:
            j.reset_current_position(
                self.np_random.uniform(low=-0.1, high=0.1), 0)
        self.feet = [self.parts[f] for f in self.foot_list]
        self.feet_contact = np.array(
            [0.0 for f in self.foot_list], dtype=np.float32)
        self.scene.actor_introduce(self)
        self.initial_z = None

    def move_robot(self, init_x, init_y, init_z):
        "Used by multiplayer stadium to move sideways, to another running lane."
        self.cpp_robot.query_position()
        pose = self.cpp_robot.root_part.pose()
        # Works because robot loads around (0,0,0), and some robots have z != 0 that is left intact
        pose.move_xyz(init_x, init_y, init_z)
        self.cpp_robot.set_pose(pose)
        self.start_pos_x, self.start_pos_y, self.start_pos_z = init_x, init_y, init_z

    def apply_action(self, action):
        assert(np.isfinite(action).all())
        for a, j in zip(action, self.ordered_joints):
            j.set_relative_servo_target(a, 0.7, 0.4)

    def calc_state(self):
        jointStates = [s[0] for s in p.getJointStates(
            self.yamax, range(8, self.num_joints + 8))]
        hipState = p.getLinkState(self.yamax, 9)
        euler = p.getEulerFromQuaternion(hipState[1])
        return jointStates + list(euler)

    def get_position(self):
        self.cpp_robot.query_position()
        pose = self.cpp_robot.root_part.pose()
        return pose

    foot_collision_cost = -0.1
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
        return (getRoll(self.left_leg), getRoll(self.right_leg))

    def _step(self, action):
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.apply_action(action)
            self.scene.global_step()

        state = self.calc_state()
        x, y, z = self.get_position()
        euler = state[self.num_joints:self.num_joints+3]

        c = [math.cos(a / 2) for a in euler]
        s = [math.sin(a / 2) for a in euler]
        axisAngle = 2 * math.acos(reduce(mul, c) - reduce(mul, s))
        done = axisAngle > self.fail_threshold or x > self.success_x_threshold
        feetCollisionCost = self.calc_feet_collision_cost()
        lr, ll = self.get_legs_orientation()
        legError = - 0.1 * (lr - ll) ** 2
        Or, Op, Oy = euler
        reward = -0.01 * (Or**2 + Op**2 + 3*Oy**2 + 1) * (3*y**2 + 1) + \
            feetCollisionCost + legError - (self._last_x - x)
        if axisAngle > self.fail_threshold:
            reward = -1
        elif x > self.success_x_threshold:
            reward = 1

        self._last_x = x
        return np.array(state), reward, done, {}

    def camera_adjust(self):
        # self.camera_dramatic()
        self.camera_simple_follow()

    def camera_simple_follow(self):
        x, y, z = self.body_xyz
        self.camera_x = 0.98*self.camera_x + (1-0.98)*x
        self.camera.move_and_look_at(self.camera_x, y-2.0, 1.4, x, y, 1.0)

    def camera_dramatic(self):
        pose = self.robot_body.pose()
        speed = self.robot_body.speed()
        x, y, z = pose.xyz()
        if 1:
            camx, camy, camz = speed[0], speed[1], 2.2
        else:
            camx, camy, camz = self.walk_target_x - x, self.walk_target_y - y, 2.2

        n = np.linalg.norm([camx, camy])
        if n > 2.0 and self.frame > 50:
            self.camera_follow = 1
        if n < 0.5:
            self.camera_follow = 0
        if self.camera_follow:
            camx /= 0.1 + n
            camx *= 2.2
            camy /= 0.1 + n
            camy *= 2.8
            if self.frame < 1000:
                camx *= -1
                camy *= -1
            camx += x
            camy += y
            camz = 1.8
        else:
            camx = x
            camy = y + 4.3
            camz = 2.2
        # print("%05i" % self.frame, self.camera_follow, camy)
        smoothness = 0.97
        self.camera_x = smoothness*self.camera_x + (1-smoothness)*camx
        self.camera_y = smoothness*self.camera_y + (1-smoothness)*camy
        self.camera_z = smoothness*self.camera_z + (1-smoothness)*camz
        self.camera.move_and_look_at(
            self.camera_x, self.camera_y, self.camera_z, x, y, 0.6)
