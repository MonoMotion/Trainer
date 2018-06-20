import math

import pybullet
from pybullet_envs.bullet import bullet_client
import pybullet_data

from urdf_parser_py.urdf import URDF

from Adafruit_PCA9685 import PCA9685
from mpu6050 import mpu6050

class Humanoid(object):
    def __init__(self, urdf, bullet_client, real=False, time_step=0.01, servo_angular_speed=0.14, i2c_address=0x40, i2c_busnum=2, i2c_imu_address=0x68, servo_min=150, servo_max=600):
        self.is_real     = real
        self.servo_angular_speed = servo_angular_speed
        self.time_step = time_step
        self.urdf = URDF.from_xml_file(self._urdf_path)

        self._angular_velocity_limit = math.pi / (self.servo_angular_speed * 3)
        self._urdf_path = urdf
        self._pybullet = bullet_client

        self.num_joints = len(self.urdf.joints) - 8 # HARDCODED: 8

        if self.is_real:
            self.pwm = PCA9685(i2c_address, i2c_busnum)
            self.pwm.set_pwm_freq(60)
            self.imu = mpu6050(i2c_imu_address)

        self.reset()

    def _set_servo(self, channel, rad):
        pulse = (rad + math.pi / 2) / math.pi * (self.servo_max - self.servo_min) + self.servo_min
        pulse_length = 1000000 // 60 // 4096   # 1,000,000 us per second
        pulse = pulse * 100 // pulse_length
        pwm.set_pwm(channel, 0, pulse)

    def step(self):
        if self.real:
            return

        self._pybullet.stepSimulation()

    def reset(self):
        self._pybullet.resetSimulation()
        self._pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = self._pybullet.loadURDF("plane.urdf")
        self.robot_id = self._pybullet.loadURDF(self._urdf_path, [0,0,0], flags=self._pybullet.URDF_USE_SELF_COLLISION)
        h = self._pybullet.getLinkState(self.robot_id, 19)[0][2] # HARDCODED!!
        self._pybullet.resetBasePositionAndOrientation(self.robot_id, [0,0,-h + 0.01], [0,0,0,1]) # HARDCODED: 0.01
        self.num_joints = self._pybullet.getNumJoints(self.robot_id) - 8 # HARDCODED: 8

        self._pybullet.setGravity(0,0, -9.79)
        self._pybullet.setTimeStep(self.time_step)
        self._pybullet.setRealTimeSimulation(0) # HARDCODED: 8

        self._fixed_joints = [0] * 8 # HARDCODED: 8
        self.set_joint_states([0] * self.num_joints)

    def get_joint_states(self):
        return [s[0] for s in self._pybullet.getJointStates(self.robot_id, range(8, self.num_joints + 8))] # HARDCODED: 8

    def get_rotation(self):
        if self.is_real:
            gyro = self.imu.get_gyro_data()
            return [gyro['x'], gyro['y'], gyro['z']]
        else:
            hip_state = self._pybullet.getLinkState(self.robot_id, 9)
            euler = self._pybullet.getEulerFromQuaternion(hip_state[1])
            return list(euler)

    def get_position(self):
        if self.is_real:
            raise RuntimeError("Cannot get position of real robot")
        else:
            pos, _ = self._pybullet.getBasePositionAndOrientation(self.robot_id)
            return pos

    def set_joint_state(self, idx, pos):
        if self.is_real:
            self._set_servo(idx, pos)
        else:
            self._pybullet.setJointMotorControl2(self.robot_id, 8 + idx, self._pybullet.POSITION_CONTROL, targetPosition=pos, maxVelocity=self._angular_velocity_limit) # HARDCODED: 8

    def set_joint_states(self, pos_array):
        for (idx, pos) in enumerate(pos_array):
            self.set_joint_state(idx, pos)

    def count_unpermitted_contacts(self):
        if self.is_real:
            raise RuntimeError("Cannot count contacts of real robot")
        else:
            contacts = self._pybullet.getContactPoints(bodyA=self.robot_id)
            num_valid = sum(((contact[1] == self.plane_id and contact[3] == -1) and (contact[2] == self.robot_id and (contact[4] == 19 or contact[4] == 14))) or ((contact[2] == self.plane_id and contact[4] == -1) and (contact[1] == self.robot_id and (contact[3] == 19 or contact[3] == 14))) for contact in contacts)
            return len(contacts) - num_valid

    def get_legs_orientation(self):
        if self.is_real:
            raise RuntimeError("Cannot measure rotation of legs of real robot")
        else:
            def get_roll(lidx):
                orientation = self._pybullet.getLinkState(self.robot_id, lidx)[1]
                euler = self._pybullet.getEulerFromQuaternion(orientation)
                return euler[0] # roll
            return (get_roll(12), get_roll(17))