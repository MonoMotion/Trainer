import json
from itertools import islice

import numpy as np
import quaternion

from .utils import pairwise, dictzip

class MotionIterator(object):
    def __init__(self, motion_data):
        self.loop_mode = motion_data['loop']
        self.frames = motion_data['frames']
        self.frames_iter = self.make_frames_iter()
        self.tp_offset = 0
        self.last_tp = self.frames[-1]['timepoint']

    def make_frames_iter(self):
        return iter(self.frames)

    def __iter__(self):
        iterator = self
        iterator.frames_iter = iterator.make_frames_iter()
        iterator.tp_offset = 0
        return iterator

    def __next__(self):
        try:
            frame = next(self.frames_iter)
            return self.tp_offset + frame['timepoint'], frame['position'], frame['effector']
        except StopIteration:
            if self.loop_mode == 'wrap':
                self.frames_iter = islice(self.make_frames_iter(), 1, None)
                self.tp_offset += self.last_tp
                return next(self)
            elif self.loop_mode == 'none':
                raise StopIteration
            else:
                raise NotImplementedError('Unsupported loop mode "{}"'.format(self.loop_mode))

def simple_interp(t, t1, t2, x1, x2):
    return (t1 - t) / (t2 - t1) * (x2 - x1) + x1

class LocationTargetValue(object):
    def __init__(self, space, weight, value):
        self.space = space
        self.weight = weight
        self.value = value

    def interp(self, t, t1, t2, target):
        assert self.space == target.space
        weight = simple_interp(t, t1, t2, self.weight, target.weight)
        value = simple_interp(t, t1, t2, self.value, target.value)
        return LocationTargetValue(self.space, weight, value)

class RotationTargetValue(object):
    def __init__(self, space, weight, value):
        self.space = space
        self.weight = weight
        assert isinstance(value, np.quaternion)
        self.value = value

    def interp(self, t, t1, t2, target):
        assert self.space == target.space
        weight = simple_interp(t, t1, t2, self.weight, target.weight)
        value = quaternion.slerp(self.value, target.value, t1, t2, t)
        return RotationTargetValue(self.space, weight, value)

class EffectorTarget(object):
    def __init__(self, location, rotation):
        self.location = location
        self.rotation = rotation

    def interp(self, t, t1, t2, target):
        if self.location is not None:
            location = self.location.interp(target.location)
        else:
            assert target.location is not None
            location = None

        if self.rotation is not None:
            rotation = self.rotation.interp(target.rotation)
        else:
            assert target.rotation is not None
            rotation = None

        return EffectorTarget(location, rotation)


def make_effector_target(obj):
        location = obj.get('location')
        location_target = LocationTargetValue(location['space'], location['weight'], location['value']) if location is not None else None

        rotation = obj.get('rotation')
        rotation_target = RotationTargetValue(rotation['space'], rotation['weight'], np.quaternion(*rotation['value'])) if rotation is not None else None

        return EffectorTarget(location_target, rotation_target)

def get_frame_at(t, motion_iter):
    """
        t: float
        motion_iter: MotionIterator
    """

    def calc_gap(t, t1, t2, p1, p2, e1, e2):
        positions = {j: simple_interp(t, t1, t2, p1, p2) for j, (p1, p2) in dictzip(p1, p2)}

        def conv(eff):
            return {l: make_effector_target(e1) for l, e in eff}

        effectors = {l: e1.interp(t, t1, t2, e2) for l, (e1, e2) in dictzip(conv(e1), conv(e2))}

        return positions, effectors

    return next(calc_gap(t, t1, t2, p1, p2, e1, e2) for (t1, p1, e1), (t2, p2, e2) in pairwise(motion_iter) if t1 <= t < t2)

