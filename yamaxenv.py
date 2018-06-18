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
from functools import reduce
from operator import mul
from pkg_resources import parse_version

import pybullet
from pybullet_envs.bullet import bullet_client

from humanoid import Humanoid

class YamaXEnv(gym.Env):
  metadata = {
    'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second' : 10
  }

  def __init__(self, logdir, renders=True, robotUrdf="yamax.urdf", frame_delay=0, render_size=(320, 240), cam_dist=0.75, cam_yaw=75, cam_pitch=-15):
      # start the bullet physics server
    if logdir:
        self._reward_log_file = open(os.path.join(logdir, 'log.csv'), 'wt')
        self._logger = csv.DictWriter(self._reward_log_file, fieldnames=('time_elapsed', 'reward_sum', 'final_reward', 'maximum_leg_error', 'num_timesteps', 'distance_sum', 'final_distance', 'unperm_sum'))
    else:
        self._reward_log_file = None
        self._logger = None

    self._renders = renders
    self._updateDelay = frame_delay
    self._cam_dist = cam_dist
    self._cam_yaw = cam_yaw
    self._cam_pitch = cam_pitch
    self._rendering_size = render_size

    if self._renders:
        self._pybullet = bullet_client.BulletClient(
            connection_mode=pybullet.GUI, options=f"--width={self._rendering_size[0]} --height={self._rendering_size[1]}")
    else:
        self._pybullet = bullet_client.BulletClient()

    # self._pybullet.configureDebugVisualizer(self._pybullet.COV_ENABLE_WIREFRAME, 1)
    self._pybullet.configureDebugVisualizer(self._pybullet.COV_ENABLE_GUI, 0)
    self._pybullet.configureDebugVisualizer(self._pybullet.COV_ENABLE_MOUSE_PICKING, 0)
    self._pybullet.resetDebugVisualizerCamera(self._cam_dist + 1, self._cam_yaw, self._cam_pitch, [0,0,0])

    self.robot = Humanoid(urdf=robotUrdf, bullet_client=self._pybullet, render=renders)

    action_high = np.array([50 * math.pi / 180] * self.robot.num_joints)
    observation_high = np.concatenate((action_high, [np.finfo(np.float32).max] * 3))

    self.action_space = spaces.Box(-action_high, action_high)
    self.observation_space = spaces.Box(-observation_high, observation_high)

    self.fail_threshold = 45 * math.pi / 180
    self.success_x_threshold = 3
    self._seed(0)
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
    self.robot.step()

    if self._updateDelay:
        time.sleep(self._updateDelay)
    self._updateState()
    joint_states = self.state[:self.robot.num_joints]

    applied = [a + da for (a, da) in zip(joint_states, action)]
    self.robot.set_joint_states(applied)
    for _ in range(5):
        self.robot.step()

    self._updateState()
    x, y, z = self.robot.get_position()
    euler = self.state[self.robot.num_joints:self.robot.num_joints+3]

    c = [math.cos(a / 2) for a in euler]
    s = [math.sin(a / 2) for a in euler]
    axisAngle = 2 * math.acos(reduce(mul, c) - reduce(mul, s))
    done = axisAngle > self.fail_threshold or x > self.success_x_threshold
    numUnpermittedContact = self.robot.count_unpermitted_contacts()
    lr, ll = self.robot.get_legs_orientation()
    legError = - 0.1 * (lr - ll) ** 2
    Or, Op, Oy = euler
    reward = -0.01 * (Or**2 + Op**2 + 3*Oy**2 + 1) * (3*y**2 + 1) - 0.1 * numUnpermittedContact + legError - (self._last_x - x)
    if axisAngle > self.fail_threshold:
      reward = -1
    elif x > self.success_x_threshold:
      reward = 1

    self._ep_rewards.append(reward)
    self._ep_legs.append(legError)
    self._ep_unperms.append(numUnpermittedContact)
    if done:
        if self._logger:
            eprew = sum(self._ep_rewards)
            eplen = len(self._ep_rewards)
            epinfo = {"reward_sum": round(eprew, 6), "num_timesteps": eplen, "time_elapsed": round(time.time() - self._tstart, 6), "final_reward": reward, "final_distance": x, "distance_sum": x - self._last_ep_x, "maximum_leg_error": max(self._ep_legs), "unperm_sum": sum(self._ep_unperms)}
            self._last_ep_x = x
            self._logger.writerow(epinfo)
            self._reward_log_file.flush()

    self._last_x = x
    return np.array(self.state), reward, done, {}

  def _reset(self):
#    print("-----------reset simulation---------------")
    self.robot.reset()

    self._updateState()

    self._last_x = 0
    self._ep_rewards = []
    self._ep_legs = []
    self._ep_unperms = []
    return np.array(self.state)

  def _updateState(self):
      self.state = self.robot.get_joint_states() + self.robot.get_rotation()

  def _render(self, mode='human', close=False):
      if mode=="human" and not self._renders:
          raise RuntimeError("Supplied mode=='human' but connected as p.DIRECT")
      if mode != "rgb_array":
          return np.array([])

      base_pos=self.robot.get_position()

      view_matrix = self._pybullet.computeViewMatrixFromYawPitchRoll(
          cameraTargetPosition=base_pos,
          distance=self._cam_dist,
          yaw=self._cam_yaw,
          pitch=self._cam_pitch,
          roll=0,
          upAxisIndex=2)
      width, height = self._rendering_size
      proj_matrix = self._pybullet.computeProjectionMatrixFOV(
          fov=60, aspect=float(width)/height,
          nearVal=0.1, farVal=100.0)
      (_, _, px, _, _) = self._pybullet.getCameraImage(
      width=width, height=height, viewMatrix=view_matrix,
          projectionMatrix=proj_matrix,
          renderer=self._pybullet.ER_BULLET_HARDWARE_OPENGL
          )
      if len(px) != height:
          px = np.reshape(px, (height, width, 4)).astype('uint8')
      rgb_array = np.array(px)
      rgb_array = rgb_array[:, :, :3]
      return rgb_array

  if parse_version(gym.__version__)>=parse_version('0.9.6'):
    render = _render
    reset = _reset
    seed = _seed
    step = _step
