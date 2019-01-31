from collections.abc import MutableSequence
import numpy as np


class LoopedWeights(MutableSequence):
    def __init__(self, num_frames: int, num_joints: int):
        self.weights = np.zeros(shape=(num_frames - 1, num_joints))
        self.size = num_frames

    def calc_index(self, key: int) -> int:
        idx: int = key % self.size
        if idx == self.size - 1:
            return 0
        else:
            return idx

    def __setitem__(self, key: int, value: np.ndarray):
        self.weights[self.calc_index(key)] = value

    def __getitem__(self, key: int) -> np.ndarray:
        return self.weights[self.calc_index(key)]

    def __delitem__(self, key: int):
        del self.weights[self.calc_index(key)]

    def insert(self, key: int, value: np.ndarray):
        raise NotImplementedError()

    def __len__(self):
        return self.size
