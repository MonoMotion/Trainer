import numpy as np
from typing import Dict, Optional, List, Callable, Tuple
import dataclasses
from logging import getLogger
import math
from concurrent import futures
import threading

from nevergrad.optimization import optimizerlib
from nevergrad.instrumentation import InstrumentedFunction
from nevergrad.instrumentation.variables import Gaussian

from .simulation import apply_joints
from .evaluation import calc_reward
from .silver_bullet import Scene, Robot
from .silver_bullet.scene import SavedState
from .utils import try_get_pre_positions

import flom

log = getLogger(__name__)


def apply_weights(positions, weights):
    # sort is required because frame order is nondeterministic
    return {k: v + w for w, (k, v) in zip(weights, sorted(positions.items()))}


@dataclasses.dataclass
class StateWithJoints:
    saved_state: SavedState
    joint_torques: Dict[str, float]

    def restore(self, scene: Scene, robot: Robot):
        scene.restore_state(self.saved_state)
        for name, force in self.joint_torques.items():
            robot.set_joint_torque(name, force)

    @staticmethod
    def save(scene: Scene, robot: Robot):
        torques = {name: robot.joint_state(name).applied_torque for name in robot.joints.keys()}
        return StateWithJoints(scene.save_state(), torques)


@dataclasses.dataclass
class Env:
    scene: Scene
    robot: Robot

    state: Optional[StateWithJoints] = None

    def __post_init__(self):
        self.save()

    def save(self):
        self.state = StateWithJoints.save(self.scene, self.robot)

    def restore(self):
        self.state.restore(self.scene, self.robot)


def train_chunk(motion: flom.Motion, envs: List[Env], start: float, init_weights: np.ndarray, *, algorithm: str = 'OnePlusOne', num_iteration: int = 1000, weight_factor: float = 0.01, stddev: float = 1, **kwargs):
    weight_shape = np.array(init_weights).shape

    thread_envs = {}  # type: Dict[int, Env]

    def step(weights, env: Env = None):
        env = env or thread_envs[threading.get_ident()]

        env.restore()

        reward_sum = 0
        start_ts = env.scene.ts

        pre_positions = try_get_pre_positions(env.scene, motion, start=start)

        for init_weight, frame_weight in zip(init_weights, weights):
            frame = motion.frame_at(start + env.scene.ts - start_ts)

            frame.positions = apply_weights(
                frame.positions, init_weight + frame_weight * weight_factor)
            apply_joints(env.robot, frame.positions)

            env.scene.step()

            reward_sum += calc_reward(motion, env.robot, frame, pre_positions, **kwargs)

            pre_positions = frame.positions

        return -reward_sum

    used_idx = 0
    lock = threading.Lock()
    def register_thread():
        nonlocal used_idx, thread_envs
        with lock:
            thread_envs[threading.get_ident()] = envs[used_idx]
            used_idx += 1

    weights_param = Gaussian(mean=0, std=stddev, shape=weight_shape)
    inst_step = InstrumentedFunction(step, weights_param)
    optimizer = optimizerlib.registry[algorithm](
        dimension=inst_step.dimension, budget=num_iteration, num_workers=len(envs))

    with futures.ThreadPoolExecutor(max_workers=optimizer.num_workers, initializer=register_thread) as executor:
        recommendation = optimizer.optimize(inst_step, executor=executor)
    weights = np.reshape(recommendation, weight_shape)

    for e in envs:
        reward = step(weights, e)
        e.save()

    return reward, weights * weight_factor


def train(motion: flom.Motion, make_scene: Callable[[int], Tuple[Scene, Robot]], *, num_workers: int = 5, chunk_length: int = 3, num_chunk: Optional[int] = None, **kwargs):
    envs = [Env(*make_scene(i)) for i in range(num_workers)]
    first_env = envs[0]

    chunk_duration = first_env.scene.dt * chunk_length

    if num_chunk is None:
        num_chunk = math.ceil(motion.length() / chunk_duration)

    total_length = chunk_duration * num_chunk
    log.info(f"chunk duration: {chunk_duration} s")
    log.info(f"motion length: {motion.length()} s")
    log.info(f"total length of train: {total_length} s")

    if total_length < motion.length():
        log.warning(f"A total length to train is shorter than the length of motion")

    num_frames = int(motion.length() / first_env.scene.dt)
    num_joints = len(list(motion.joint_names()))  # TODO: Call len() directly
    weights = np.zeros(shape=(num_frames, num_joints))
    log.info(f"shape of weights: {weights.shape}")
    log.debug(f"kwargs: {kwargs}")

    for chunk_idx in range(num_chunk):
        start = chunk_idx * chunk_duration
        start_idx = chunk_idx * chunk_length % num_frames

        r = range(start_idx, start_idx + chunk_length)
        in_weights = [weights[i % num_frames] for i in r]
        log.info(f"start training chunk {chunk_idx} ({start}~)")
        reward, out_weights = train_chunk(motion, envs, start, in_weights, **kwargs)
        for i, w in zip(r, out_weights):
            weights[i % num_frames] = w

        log.info(f"chunk {chunk_idx}: {reward}")

    # Use copy ctor after DeepL2/flom-py#23
    types = {n: motion.effector_type(n) for n in motion.effector_names()}
    new_motion = flom.Motion(set(motion.joint_names()), types, motion.model_id())
    new_motion.set_loop(motion.loop())
    for name in motion.effector_names():
        new_motion.set_effector_weight(name, motion.effector_weight(name))

    for i, frame_weight in enumerate(weights):
        t = i * first_env.scene.dt
        new_frame = motion.frame_at(t)
        new_frame.positions = apply_weights(new_frame.positions, frame_weight)
        new_motion.insert_keyframe(t, new_frame)
    return new_motion
