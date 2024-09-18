import numpy as np
import random


class Memory:
    epsilon = 1e-5
    alpha = 0.8  # Controls how much prioritization is used
    beta = 0.3
    beta_increment_per_sampling = 0.0005

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        priority = (np.abs(error) + self.epsilon) ** self.alpha
        return priority

    def _get_reward_factor(self, reward):
        reward_factor = (reward + 1e-5) ** (self.alpha * 0.5)  # Scale reward influence
        return reward_factor

    def add(self, error, reward, sample):
        # Prioritize based on a combination of TD-error and reward
        priority = self._get_priority(error)
        reward_factor = self._get_reward_factor(reward)
        final_priority = priority * reward_factor

        self.tree.add(final_priority, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)

            (idx, priority, data) = self.tree.get(s)

            priorities.append(priority)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error, reward):
        priority = self._get_priority(error)
        reward_factor = self._get_reward_factor(reward)
        final_priority = priority * reward_factor
        self.tree.update(idx, final_priority)


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity  # Set the capacity of the SumTree
        self.tree = np.zeros(2 * capacity - 1)   # Initialize the tree with zeros
        self.data = np.zeros(capacity, dtype=object)  # Initialize the data array
        self.write = 0  # Initialize the write pointer
        self.n_entries = 0

    def _propagate(self, idx, change):
        change = np.real(change)
        parent = (idx - 1) // 2  # Calculate the parent index

        self.tree[parent] += change  # Update the parent's value

        if parent != 0:
            self._propagate(parent, change)  # Recursively propagate the change upwards

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):  # If we're at a leaf node
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]  # Return the total sum (root of the tree)

    def add(self, priority, data):
        priority = abs(np.real(priority))  # to ensure priority is real and non-negative value
        idx = self.write + self.capacity - 1  # Calculate the index to write to

        self.data[self.write] = data
        self.update(idx, priority)

        self.write += 1  # Move the write pointer
        if self.write >= self.capacity:  # If we've reached the end
            self.write = 0  # Reset the write pointer (circular buffer)

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, priority):
        priority = abs(np.real(priority))  # to ensure priority is real and non-negative value
        change = priority - self.tree[idx]  # Calculate the change in priority

        self.tree[idx] = priority  # Update the priority
        self._propagate(idx, change)  # Propagate the change upwards

    def get(self, s):
        idx = self._retrieve(0, s)  # Get the index for the given sum
        dataIdx = idx - self.capacity + 1  # Calculate the data index

        return idx, self.tree[idx], self.data[dataIdx]  # Return index, priority, and data
