import math
import numpy as np
import quaternion

from evostra import EvolutionStrategy
from .simulation import reset, apply_joints
from .utils import select_location, select_rotation
from .silver_bullet import Scene

import flom

def apply_weights(positions, weights):
    # sort is required because frame order is nondeterministic
    return {k: v + w for w, (k, v) in zip(weights, sorted(positions.items()))}

def calc_reward(motion, robot, frame):
    diff = 0
    for name, effector in frame.effectors.items():
        pose = robot.link_state(name).pose
        root_pose = robot.link_state(robot.root_link).pose
        weight = motion.effector_weight(name)
        ty = motion.effector_type(name)
        if effector.location:
            target = select_location(ty.location, effector.location.vector, root_pose)
            diff += np.linalg.norm(pose.vector - np.array(target)) ** 2 * weight.location
        if effector.rotation:
            target = select_rotation(ty.rotation, effector.rotation.quaternion, root_pose)
            quat1 = np.quaternion(*target)
            quat2 = np.quaternion(*pose.quaternion)
            diff += quaternion.rotation_intrinsic_distance(quat1, quat2) ** 2 * weight.rotation
    k = 1
    normalized = k * diff / len(frame.effectors)
    return - math.exp(normalized) + 1


def train(motion, robot_file, timestep=0.0165/8, frame_skip=8):
    scene = Scene(timestep, frame_skip)

    def step(weights, enable_render=False):
        robot = reset(scene, robot_file)

        reward_sum = 0
        for frame_weight in weights:
            scene.step()

            frame = motion.frame_at(scene.ts)

            reward_sum += calc_reward(motion, robot, frame)

            apply_joints(robot, apply_weights(frame.positions, frame_weight))

            if enable_render:
                pass

        return reward_sum

    num_frames = int(motion.length() / scene.dt)
    num_joints = len(list(motion.joint_names()))  # TODO: Call len() directly
    weights = np.zeros(shape=(num_frames, num_joints), dtype='float')
    es = EvolutionStrategy(weights, step, population_size=20, sigma=0.1, learning_rate=0.03, decay=0.995, num_threads=1)
    es.run(1000, print_step=1)

    # Use copy ctor after DeepL2/flom-py#23
    types = {n: motion.effector_type(n) for n in motion.effector_names()}
    new_motion = flom.Motion(set(motion.joint_names()), types, motion.model_id())
    new_motion.set_loop(motion.loop())
    for name in motion.effector_names():
        new_motion.set_effector_weight(name, motion.effector_weight(name))

    for i, frame_weight in enumerate(es.get_weights()):
        t = i * scene.dt
        new_frame = motion.frame_at(t)
        new_frame.positions = apply_weights(new_frame.positions, frame_weight)
        new_motion.insert_keyframe(t, new_frame)
    return new_motion
