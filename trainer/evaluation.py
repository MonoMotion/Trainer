import numpy as np
import quaternion

import math

from .utils import select_location, select_rotation


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

