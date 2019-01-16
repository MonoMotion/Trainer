import flom

from .silver_bullet import Pose
import numpy as np

import random

def dictzip(d1, d2):
    for k, v in d1.items():
        yield k, (v, d2[k])

def select_location(ty, vec, root_pose):
    if ty == flom.CoordinateSystem.World:
        return vec
    elif ty == flom.CoordinateSystem.Local:
        pose = Pose(np.array(vec), np.array([0,0,0,0]))
        return root_pose.dot(pose).vector
    else:
        assert False  # unreachable

def select_rotation(ty, quat, root_pose):
    if ty == flom.CoordinateSystem.World:
        return quat
    elif ty == flom.CoordinateSystem.Local:
        pose = Pose(np.array([0,0,0]), np.array(quat))
        return root_pose.dot(pose).quatertion
    else:
        assert False  # unreachable

def add_noise(motion, randomness):
    for t, frame in motion.keyframes():
        new_frame = frame.get()
        positions = {
            k: v + random.random() * randomness
            for k, v in new_frame.positions.items()
        }
        new_frame.positions = positions
        frame.set(new_frame)
