import numpy as np
from typing import Dict
import dataclasses

from evostra import EvolutionStrategy
from .simulation import apply_joints
from .evaluation import calc_reward
from .silver_bullet import Scene, Robot
from .silver_bullet.scene import SavedState

import flom


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


def train_chunk(scene: Scene, motion: flom.Motion, robot: Robot, start: float, init_weights: np.ndarray, init_state: StateWithJoints, num_iteration: int = 100, weight_factor: float = 0.01):
    def step(weights):
        init_state.restore(scene, robot)

        reward_sum = 0
        start_ts = scene.ts
        for frame_weight in weights:
            frame = motion.frame_at(start + scene.ts - start_ts)

            reward_sum += calc_reward(motion, robot, frame)

            apply_joints(robot, apply_weights(frame.positions, frame_weight * weight_factor))

            scene.step()

        return reward_sum

    es = EvolutionStrategy(init_weights, step, population_size=20, sigma=0.1,
                           learning_rate=0.03, decay=0.995, num_threads=1)
    es.run(num_iteration, print_step=1)

    weights = es.get_weights()
    reward = step(weights)

    state = StateWithJoints.save(scene, robot)
    return reward, weights, state


def train(scene, motion, robot, chunk_length=3, num_iteration=500, num_chunk=100, weight_factor=0.01):
    chunk_duration = scene.dt * chunk_length

    num_frames = int(motion.length() / scene.dt)
    num_joints = len(list(motion.joint_names()))  # TODO: Call len() directly
    weights = np.zeros(shape=(num_frames, num_joints))

    last_state = StateWithJoints.save(scene, robot)
    for chunk_idx in range(num_chunk):
        start = chunk_idx * chunk_duration
        start_idx = chunk_idx * chunk_length % num_frames

        r = range(start_idx, start_idx + chunk_length)
        in_weights = [weights[i % num_frames] for i in r]
        print("start training chunk {} ({}~)".format(chunk_idx, start))
        reward, out_weights, last_state = train_chunk(
            scene, motion, robot, start, in_weights, last_state, num_iteration, weight_factor)
        for i, w in zip(r, out_weights):
            weights[i % num_frames] = w

        print("chunk {}: {}".format(chunk_idx, reward))

    # Use copy ctor after DeepL2/flom-py#23
    types = {n: motion.effector_type(n) for n in motion.effector_names()}
    new_motion = flom.Motion(set(motion.joint_names()), types, motion.model_id())
    new_motion.set_loop(motion.loop())
    for name in motion.effector_names():
        new_motion.set_effector_weight(name, motion.effector_weight(name))

    for i, frame_weight in enumerate(weights):
        t = i * scene.dt
        new_frame = motion.frame_at(t)
        new_frame.positions = apply_weights(new_frame.positions, frame_weight * weight_factor)
        new_motion.insert_keyframe(t, new_frame)
    return new_motion
