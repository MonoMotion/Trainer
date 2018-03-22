import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import csv
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math
import time
import subprocess
import pybullet as p
import pybullet_data
from functools import reduce
from operator import mul
import matplotlib.pyplot as plt
from pkg_resources import parse_version

class YamaXEnv(gym.Env):
  metadata = {
    'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second' : 50
  }

  def __init__(self, logdir, renders=True, robotUrdf="yamax.urdf"):
      # start the bullet physics server
    if logdir:
        self._reward_log_file = open(os.path.join(logdir, 'log.csv'), 'wt')
        self._logger = csv.DictWriter(self._reward_log_file, fieldnames=('time_elapsed', 'reward_sum', 'final_reward', 'maximum_leg_error', 'num_timesteps', 'distance_sum', 'final_distance'))
    else:
        self._reward_log_file = None
        self._logger = None

    self._renders = renders
    self._urdf = robotUrdf
    if (renders):
        p.connect(p.GUI)
    else:
    	p.connect(p.DIRECT)
    self._cam_dist = 3
    self._cam_yaw = 0
    self._cam_pitch = -30
    self._render_width =320
    self._render_height = 240

    self.num_joints = 12 # joint idx 8 ~ is needed
    action_high = np.array([50 * math.pi / 180] * self.num_joints)
    observation_high = np.concatenate((action_high, [np.finfo(np.float32).max] * 6))

    self.action_space = spaces.Box(-action_high, action_high)
    self.observation_space = spaces.Box(-observation_high, observation_high)

    servo_angular_speed = 0.14
    self._angular_velocity_limit = math.pi / (servo_angular_speed * 3)
    self.fail_threshold = 45 * math.pi / 180
    self.success_x_threshold = 3
    self._seed()
#    self.reset()
    self.viewer = None
    self._configure()

    self._last_ep_x = 0
    self._tstart = time.time()

  def _configure(self, display=None):
    self.display = display

  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def _step(self, action):
    p.stepSimulation()
    # time.sleep(self.timeStep)
    self._updateState()
    jointStates = self.state[:self.num_joints]

    jointStatesApplied = [a + da for (a, da) in zip(jointStates, action)]
    self._setJointMotorControlArrayWithLimit(targetPositions=jointStatesApplied, maxVelocity=self._angular_velocity_limit)
    for _ in range(5):
        p.stepSimulation()

    self._updateState()
    x, y, z = self.state[self.num_joints:self.num_joints+3]
    euler = self.state[self.num_joints+3:]

    c = [math.cos(a / 2) for a in euler]
    s = [math.sin(a / 2) for a in euler]
    axisAngle = 2 * math.acos(reduce(mul, c) - reduce(mul, s))
    done = axisAngle > self.fail_threshold or x > self.success_x_threshold
    numUnpermittedContact = self._checkUnpermittedContacts()
    self._num_unpermitted += numUnpermittedContact
    lr, ll = self._getLegsOrientation()
    reward = -0.01 * sum([a*a for a in euler], 1) * (y*y + 1) - 0.1 * numUnpermittedContact - 0.1 * (lr - ll) ** 2 - (self._last_x - x)
    if axisAngle > self.fail_threshold:
      reward = -1
    elif x > self.success_x_threshold:
      reward = 1

    self._ep_rewards.append(reward)
    self._ep_legs.append(lr-ll)
    if done:
        if self._logger:
            eprew = sum(self._ep_rewards)
            eplen = len(self._ep_rewards)
            epinfo = {"reward_sum": round(eprew, 6), "num_timesteps": eplen, "time_elapsed": round(time.time() - self._tstart, 6), "final_reward": reward, "final_distance": x, "distance_sum": x - self._last_ep_x, "maximum_leg_error": max(self._ep_legs)}
            self._last_ep_x = x
            self._logger.writerow(epinfo)
            self._reward_log_file.flush()

    self._last_x = x
    return np.array(self.state), reward, done, {}

  def _setJointMotorControlArrayWithLimit(self, targetPositions, maxVelocity):
    for (idx, pos) in enumerate(self._fixed_joints + targetPositions):
        p.setJointMotorControl2(self.yamax, idx, p.POSITION_CONTROL, targetPosition=pos, maxVelocity=maxVelocity)

  def _checkUnpermittedContacts(self):
    contacts = p.getContactPoints(bodyA=self.yamax)
    numValid = sum(((contact[1] == self.plane and contact[3] == -1) and (contact[2] == self.yamax and (contact[4] == 19 or contact[4] == 14))) or ((contact[2] == self.plane and contact[4] == -1) and (contact[1] == self.yamax and (contact[3] == 19 or contact[3] == 14))) for contact in contacts)
    return len(contacts) - numValid

  def _getLegsOrientation(self):
    def getRoll(lidx):
        orientation = p.getLinkState(self.yamax, lidx)[1]
        euler = p.getEulerFromQuaternion(orientation)
        return euler[0] # roll
    return (getRoll(12), getRoll(17))

  def _reset(self):
#    print("-----------reset simulation---------------")
    self._num_unpermitted = 0
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    self.plane = p.loadURDF("plane.urdf")
    self.yamax = p.loadURDF(self._urdf, [0,0,0])
    h = p.getLinkState(self.yamax, 19)[0][2] # HARDCODED!!
    p.resetBasePositionAndOrientation(self.yamax, [0,0,-h + 0.01], [0,0,0,1]) # HARDCODED: 0.01
    self.timeStep = 0.01#0.01
    numJoints = p.getNumJoints(self.yamax) - 8
    assert numJoints == self.num_joints
    p.setGravity(0,0, -9.79)
    p.setTimeStep(self.timeStep)
    p.setRealTimeSimulation(0)

    self._fixed_joints = [0] * 8
    initialJointAngles = [0] * self.num_joints # self.np_random.uniform(low=-0.5, high=0.5, size=(self.num_joints,))
    self._setJointMotorControlArrayWithLimit(targetPositions=initialJointAngles, maxVelocity=self._angular_velocity_limit)

    self._updateState()

    self._last_x = 0
    self._ep_rewards = []
    self._ep_legs = []
    return np.array(self.state)

  def _updateState(self):
      jointStates = [s[0] for s in p.getJointStates(self.yamax, range(8, self.num_joints + 8))]
      (x, y, z), _ = p.getBasePositionAndOrientation(self.yamax)
      hipState = p.getLinkState(self.yamax, 9)
      euler = p.getEulerFromQuaternion(hipState[1])
      self.state = jointStates + [x, y, z] + list(euler)

  def _render(self, mode='human', close=False):
      if mode=="human" and not self._renders:
          raise RuntimeError("Supplied mode=='human' but connected as p.DIRECT")
      if mode != "rgb_array":
          return np.array([])

      base_pos=[0,0,0]
      if hasattr(self,'robot'):
          if hasattr(self.robot,'body_xyz'):
              base_pos = self.robot.body_xyz

      view_matrix = p.computeViewMatrixFromYawPitchRoll(
          cameraTargetPosition=base_pos,
          distance=self._cam_dist,
          yaw=self._cam_yaw,
          pitch=self._cam_pitch,
          roll=0,
          upAxisIndex=2)
      proj_matrix = p.computeProjectionMatrixFOV(
          fov=60, aspect=float(self._render_width)/self._render_height,
          nearVal=0.1, farVal=100.0)
      (_, _, px, _, _) = p.getCameraImage(
      width=self._render_width, height=self._render_height, viewMatrix=view_matrix,
          projectionMatrix=proj_matrix,
          renderer=p.ER_BULLET_HARDWARE_OPENGL
          )
      if len(px) != self._render_height:
          px = np.reshape(px, (self._render_height, self._render_width, 4)).astype('uint8')
      rgb_array = np.array(px)
      rgb_array = rgb_array[:, :, :3]
      return rgb_array

  if parse_version(gym.__version__)>=parse_version('0.9.6'):
    render = _render
    reset = _reset
    seed = _seed
    step = _step
