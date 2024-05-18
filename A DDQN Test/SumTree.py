import numpy


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.data = numpy.zeros(capacity, dtype=object)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return idx, self.tree[idx], self.data[dataIdx]

# import numpy as np
# from collections import namedtuple
#
# Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
#
#
# class SumTree:
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.data = np.zeros(capacity, dtype=object)
#         self.priorities = np.zeros(2 * capacity - 1)
#         self.write_idx = 0
#
#     def _propagate(self, idx, change):
#         parent = (idx - 1) // 2
#         self.priorities[parent] += change
#
#         if parent != 0:
#             self._propagate(parent, change)
#
#     def _retrieve(self, idx, s):
#         left = 2 * idx + 1
#         right = left + 1
#
#         if left >= len(self.priorities):
#             return idx
#
#         if s <= self.priorities[left]:
#             return self._retrieve(left, s)
#         else:
#             return self._retrieve(right, s - self.priorities[left])
#
#     def total(self):
#         return self.priorities[0]  # root node is the total priority
#
#     def add(self, priority, data):
#         idx = self.write_idx + self.capacity - 1
#
#         self.data[self.write_idx] = data
#         self._propagate(idx, priority)
#
#         self.write_idx += 1
#         if self.write_idx >= self.capacity:
#             self.write_idx = 0
#
#     def update(self, idx, priority):
#         change = priority - self.tree[idx]
#
#         self.priorities[idx] = priority
#         self._propagate(idx, change)
#
#     def get(self, s):
#         idx = self._retrieve(0, s)
#         data_idx = idx - self.capacity + 1
#
#         return idx, self.priorities[idx], self.data[data_idx]
#
