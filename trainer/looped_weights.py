from collections.abc import MutableSequence
import numpy as np


class LoopedWeights(MutableSequence):
    def __init__(self, num_frames: int, num_joints: int):
        self.weights = np.zeros(shape=(num_frames - 1, num_joints))
        self.size = num_frames

    def __setitem__(self, key: int, value: np.ndarray):
        idx: int = key % self.size
        if idx == self.size - 1:
            self.weights[0] = value
        else:
            self.weights[idx] = value

    def __getitem__(self, key: int) -> np.ndarray:
        idx: int = key % self.size
        if idx == self.size - 1:
            return self.weights[0]
        else:
            return self.weights[idx]

    def __delitem__(self, key: int):
        idx: int = key % self.size
        if idx == self.size - 1:
            del self.weights[0]
        else:
            del self.weights[idx]

    def insert(self, key: int, value: np.ndarray):
        raise NotImplementedError()

    def __len__(self):
        return self.size
