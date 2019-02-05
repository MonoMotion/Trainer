import numpy as np
from typing import Dict, Optional, Callable, List
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


def train_chunk(scene: Scene, motion: flom.Motion, init_frames: List[flom.Frame], robot: Robot, start: float, init_state: StateWithJoints, *, algorithm: str = 'OnePlusOne', num_iteration: int = 1000, weight_factor: float = 0.01, stddev: float = 1, **kwargs):
    chunk_length = len(init_frames)
    num_joints = len(init_frames[0].positions)
    weight_shape = (chunk_length, num_joints)

    def step(weights):
        init_state.restore(scene, robot)

        reward_sum = 0
        start_ts = scene.ts

        pre_positions = try_get_pre_positions(scene, motion, start=start)

        for frame, frame_weight in zip(init_frames, weights):
            positions = apply_weights(
                frame.positions, frame_weight * weight_factor)
            apply_joints(robot, positions)

            scene.step()

            reward_sum += calc_reward(motion, robot, frame.effectors, positions, pre_positions, **kwargs)

            pre_positions = positions

        score = reward_sum / len(weights)
        return -score

    weights_param = Gaussian(mean=0, std=stddev, shape=weight_shape)
    inst_step = InstrumentedFunction(step, weights_param)
    optimizer = optimizerlib.registry[algorithm](
        dimension=inst_step.dimension, budget=num_iteration, num_workers=1)
    recommendation = optimizer.optimize(inst_step)
    args, _ = inst_step.convert_to_arguments(recommendation)
    raw_weights = args[0]

    score = -step(raw_weights)

    state = StateWithJoints.save(scene, robot)

    def make_frame(frame, weight):
        new_frame = frame.new_compatible_frame()
        new_frame.positions = apply_weights(frame.positions, weight)
        new_frame.effectors = frame.effectors
        return new_frame

    weights = raw_weights * weight_factor
    frames = [make_frame(frame, weight) for frame, weight in zip(init_frames, weights)]

    return score, frames, state


def copy_motion(base: flom.Motion) -> flom.Motion:
    # Use copy ctor after DeepL2/flom-py#23
    types = {n: base.effector_type(n) for n in base.effector_names()}
    new_motion = flom.Motion(set(base.joint_names()), types, base.model_id())
    new_motion.set_loop(base.loop())
    for name in base.effector_names():
        new_motion.set_effector_weight(name, base.effector_weight(name))
    for t, frame in base.keyframes():
        new_motion.insert_keyframe(t, frame.get())

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

    log.debug(f"kwargs: {kwargs}")

    out_motion = copy_motion(motion)

    init_state = StateWithJoints.save(scene, robot)

    last_state = init_state
    for chunk_idx in range(num_chunk):
        start = chunk_idx * chunk_duration
        start_idx = chunk_idx * chunk_length

        r = range(start_idx, start_idx + chunk_length)
        in_frames = [motion.frame_at(i * scene.dt) for i in r]
        log.info(f"[chunk {chunk_idx}] start training ({start}~)")
        score, out_frames, last_state = train_chunk(scene, out_motion, in_frames, robot, start, last_state, **kwargs)
        for i, frame in zip(r, out_frames):
            out_motion.insert_keyframe(i * scene.dt % motion.length(), frame)

        log.info(f"[chunk {chunk_idx}] score: {score}")

        if callback:
            callback(chunk_idx, lambda: out_motion)

    assert motion.length() == out_motion.length()

    init_state.restore(scene, robot)
    init_score = evaluate(scene, motion, robot)

    init_state.restore(scene, robot)
    final_score = evaluate(scene, out_motion, robot)

    improvement = final_score - init_score
    log.info('Done.')
    log.info(f'score: {init_score} -> {final_score} ({improvement:+f})')

    if improvement <= 0:
        log.error('Failed to train the motion')

    return out_motion
