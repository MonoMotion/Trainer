from collections.abc import MutableSequence
import numpy as np
from typing import Union


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

    def __setitem__(self, key: Union[int, slice], value: np.ndarray):
        if isinstance(key, int):
            self.weights[self.calc_index(key)] = value
        else:
            for v, idx in zip(value, range(self.size)[key]):
                self.weights[self.calc_index(idx)] = v

    def __getitem__(self, key: Union[int, slice]) -> np.ndarray:
        if isinstance(key, int):
            return self.weights[self.calc_index(key)]
        else:
            return [self.weights[self.calc_index(idx)] for idx in range(self.size)[key]]

    def __delitem__(self, key: int):
        del self.weights[self.calc_index(key)]

    def insert(self, key: int, value: np.ndarray):
        raise NotImplementedError()

    def __len__(self):
        return self.size

    @property
    def shape(self):
        shape = self.weights.shape
        return (shape[0] + 1,) + shape[1:]
