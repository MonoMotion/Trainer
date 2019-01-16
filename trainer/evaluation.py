import numpy as np
import quaternion

import math

from .utils import select_location, select_rotation
from .simulation import apply_joints


def calc_reward(motion, robot, frame, k=1, wl=1, wr=0.005):
    # TODO: Use more clear naming of hyperparameters
    diff = 0
    for name, effector in frame.effectors.items():
        pose = robot.link_state(name).pose
        root_pose = robot.link_state(robot.root_link).pose
        weight = motion.effector_weight(name)
        ty = motion.effector_type(name)
        if effector.location:
            target = select_location(ty.location, effector.location.vector, root_pose)
            diff += wl * np.linalg.norm(pose.vector - np.array(target)) ** 2 * weight.location
        if effector.rotation:
            target = select_rotation(ty.rotation, effector.rotation.quaternion, root_pose)
            quat1 = np.quaternion(*target)
            quat2 = np.quaternion(*pose.quaternion)
            diff += wr * quaternion.rotation_intrinsic_distance(quat1, quat2) ** 2 * weight.rotation
    normalized = k * diff / len(frame.effectors)
    return - math.exp(normalized) + 1


def evaluate(scene, motion, robot, loop=2, **kwargs):
    reward_sum = 0
    for t, frame in motion.frames(scene.dt):
        apply_joints(robot, frame.positions)

        scene.step()

        reward_sum += calc_reward(motion, robot, frame, **kwargs)

        if t > motion.length() * loop:
            break

    return reward_sum
