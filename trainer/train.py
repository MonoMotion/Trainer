import numpy as np
from typing import Dict, Optional, Callable
import dataclasses
from logging import getLogger
import math

from nevergrad.optimization import optimizerlib
from nevergrad.instrumentation import InstrumentedFunction
from nevergrad.instrumentation.variables import Gaussian

from .simulation import apply_joints
from .evaluation import calc_reward, evaluate
from silverbullet import Scene, Robot
from silverbullet.scene import SavedState
from .utils import try_get_pre_positions
from .looped_weights import LoopedWeights

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


def train_chunk(scene: Scene, motion: flom.Motion, robot: Robot, start: float, init_weights: np.ndarray, init_state: StateWithJoints, *, algorithm: str = 'OnePlusOne', num_iteration: int = 1000, weight_factor: float = 0.01, stddev: float = 1, **kwargs):
    weight_shape = np.array(init_weights).shape

    def step(weights):
        init_state.restore(scene, robot)

        reward_sum = 0
        start_ts = scene.ts

        pre_positions = try_get_pre_positions(scene, motion, start=start)

        for init_weight, frame_weight in zip(init_weights, weights):
            frame = motion.frame_at(start + scene.ts - start_ts)

            frame.positions = apply_weights(
                frame.positions, init_weight + frame_weight * weight_factor)
            apply_joints(robot, frame.positions)

            scene.step()

            reward_sum += calc_reward(motion, robot, frame, pre_positions, **kwargs)

            pre_positions = frame.positions

        score = reward_sum / len(weights)
        return -score

    weights_param = Gaussian(mean=0, std=stddev, shape=weight_shape)
    inst_step = InstrumentedFunction(step, weights_param)
    optimizer = optimizerlib.registry[algorithm](
        dimension=inst_step.dimension, budget=num_iteration, num_workers=1)
    recommendation = optimizer.optimize(inst_step)
    args, _ = inst_step.convert_to_arguments(recommendation)
    weights = args[0]

    score = -step(weights)

    state = StateWithJoints.save(scene, robot)
    return score, weights * weight_factor, state


def build_motion(base: flom.Motion, weights: np.ndarray, dt: float) -> flom.Motion:
    # Use copy ctor after DeepL2/flom-py#23
    types = {n: base.effector_type(n) for n in base.effector_names()}
    new_motion = flom.Motion(set(base.joint_names()), types, base.model_id())
    new_motion.set_loop(base.loop())
    for name in base.effector_names():
        new_motion.set_effector_weight(name, base.effector_weight(name))

    for i, frame_weight in enumerate(weights):
        t = i * dt
        new_frame = base.frame_at(t)
        new_frame.positions = apply_weights(new_frame.positions, frame_weight)
        new_motion.insert_keyframe(t, new_frame)

    return new_motion


Callback = Callable[[int, Callable[[], flom.Motion]], None]
def train(scene: Scene, motion: flom.Motion, robot: Robot, *, chunk_length: int = 3, num_chunk: Optional[int] = None, callback: Optional[Callback] = None, **kwargs):
    chunk_duration = scene.dt * chunk_length

    if num_chunk is None:
        num_chunk = math.ceil(motion.length() / chunk_duration)
    log.info(f'number of chunks: {num_chunk}')

    total_length = chunk_duration * num_chunk
    log.info(f"chunk duration: {chunk_duration} s")
    log.info(f"motion length: {motion.length()} s")
    log.info(f"total length of train: {total_length} s")

    if total_length < motion.length():
        log.warning(f"A total length to train is shorter than the length of motion")

    num_frames = int(motion.length() / scene.dt)
    num_joints = len(list(motion.joint_names()))  # TODO: Call len() directly
    weights = LoopedWeights(num_frames, num_joints)
    log.info(f"shape of weights: {weights.shape}")
    log.debug(f"kwargs: {kwargs}")


    init_state = StateWithJoints.save(scene, robot)

    last_state = init_state
    for chunk_idx in range(num_chunk):
        start = chunk_idx * chunk_duration
        start_idx = chunk_idx * chunk_length % num_frames

        in_weights = weights[start_idx:start_idx+chunk_length]
        log.info(f"[chunk {chunk_idx}] start training ({start}~)")
        score, out_weights, last_state = train_chunk(scene, motion, robot, start, in_weights, last_state, **kwargs)
        weights[start_idx:start_idx+chunk_length] = out_weights

        log.info(f"[chunk {chunk_idx}] score: {score}")

        if callback:
            callback(chunk_idx, lambda: build_motion(motion, weights, scene.dt))

    trained_motion = build_motion(motion, weights, scene.dt)

    init_state.restore(scene, robot)
    init_score = evaluate(scene, motion, robot)

    init_state.restore(scene, robot)
    final_score = evaluate(scene, trained_motion, robot)

    improvement = final_score - init_score
    log.info('Done.')
    log.info(f'score: {init_score} -> {final_score} ({improvement:+f})')

    if improvement <= 0:
        log.error('Failed to train the motion')

    return trained_motion
