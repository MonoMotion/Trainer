import math
import numpy as np
import quaternion
from functools import reduce
from operator import mul

from evostra import EvolutionStrategy
from .simulation import create_scene, reset, apply_joints, render
from .utils import select_location, select_rotation

import flom

def apply_weights(positions, weights):
    # sort is required because frame order is nondeterministic
    return {k: v + w for w, (k, v) in zip(weights, sorted(positions.items()))}

def calc_reward(motion, robot, parts, frame):
    robot.query_position()
    diff = 0
    for name, effector in frame.effectors.items():
        pose = parts[name].pose()
        ty = motion.effector_type(name)
        if effector.location:
            target = select_location(ty.location, effector.location.vec, robot.root_part.pose())
            diff += np.linalg.norm(pose.xyz() - np.array(target)) ** 2 * effector.location.weight
        if effector.rotation:
            target = select_rotation(ty.rotation, effector.rotation.quat, robot.root_part.pose())
            quat1 = np.quaternion(*target)
            quat2 = np.quaternion(*pose.quatertion())
            diff += quaternion.rotation_intrinsic_distance(quat1, quat2) ** 2 * effector.rotation.weight
    k = 1
    normalized = k * diff / len(frame.effectors)
    return - math.exp(normalized) + 1


def train(motion, robot_file, timestep=0.0165/8, frame_skip=8):
    scene = create_scene(timestep, frame_skip)
    robot, parts, joints = reset(scene, robot_file)
    robot.query_position()

    def step(weights, enable_render=False):
        robot, parts, joints = reset(scene, robot_file)

        reward_sum = 0
        for frame_weight in weights:
            scene.global_step()

            frame = motion.frame_at(scene.cpp_world.ts)

            reward_sum += calc_reward(motion, robot, parts, frame)

            apply_joints(joints, apply_weights(frame.positions, frame_weight))

            if enable_render:
                render(scene)

        return reward_sum

    num_frames = int(motion.length() / scene.dt)
    num_joints = len(list(motion.joint_names()))  # TODO: Call len() directly
    weights = np.zeros(shape=(num_frames, num_joints), dtype='float')
    es = EvolutionStrategy(weights, step, population_size=20, sigma=0.1, learning_rate=0.03, decay=0.995, num_threads=1)
    es.run(1000, print_step=1)

    new_motion = flom.Motion(set(motion.joint_names()), set(motion.effector_names()), motion.model_id())
    new_motion.set_loop(motion.loop())
    for i, frame_weight in enumerate(es.get_weights()):
        t = i * scene.dt
        new_frame = motion.frame_at(t)
        new_frame.positions = apply_weights(new_frame.positions, frame_weight)
        new_motion.insert_keyframe(t, new_frame)
    return new_motion
