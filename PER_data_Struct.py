import numpy as np
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = np.zeros(capacity, dtype=object)
        self.priorities = np.zeros(2 * capacity - 1)
        self.write_idx = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.priorities[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.priorities):
            return idx

        if s <= self.priorities[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.priorities[left])

    def total(self):
        return self.priorities[0]  # root node is the total priority

    def add(self, priority, data):
        idx = self.write_idx + self.capacity - 1

        self.data[self.write_idx] = data
        self._propagate(idx, priority)

        self.write_idx += 1
        if self.write_idx >= self.capacity:
            self.write_idx = 0

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.priorities[idx], self.data[data_idx]

