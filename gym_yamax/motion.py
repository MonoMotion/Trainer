import json
from itertools import islice

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
            return self.tp_offset + frame['timepoint'], frame['position']
        except StopIteration:
            if self.loop_mode == 'wrap':
                self.frames_iter = islice(self.make_frames_iter(), 1, None)
                self.tp_offset += self.last_tp
                return next(self)
            elif self.loop_mode == 'none':
                raise StopIteration
            else:
                raise NotImplementedError('Unsupported loop mode "{}"'.format(self.loop_mode))


def get_frame_at(t, motion_iter):
    """
        t: float
        motion_iter: MotionIterator
    """

    def calc_gap(td, p1, p2):
        return {j: td * (p2 - p1) + p1 for j, (p1, p2) in dictzip(p1, p2)}

    return next(calc_gap((t1 - t) / (t2 - t1), p1, p2) for (t1, p1), (t2, p2) in pairwise(motion_iter) if t1 <= t < t2)

