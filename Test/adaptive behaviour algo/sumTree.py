import numpy as np
import random


class Memory:
    epsilon = 1e-5
    alpha = 0.8  # Controls how much prioritization is used
    beta = 0.3
    beta_increment_per_sampling = 0.0005
    reward_factor_weight = 0.4

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    # def _get_priority(self, complexity, reward):
    #     # priority = (np.abs(error) + self.epsilon) ** self.alpha
    #     priority = (complexity + self.epsilon) * (reward + self.epsilon) ** self.alpha
    #     return priority

    def _get_priority(self, tcrp_value, reward):
        # Calculate a high reward factor (e.g., square root to scale down)
        # reward_factor = (np.abs(reward) + self.epsilon) ** 0.5

        # Combine TCRP with high reward factor for final priority
        # combined_priority = (tcrp_value ** self.alpha) * (reward_factor ** self.reward_factor_weight)

        priority = (tcrp_value + self.epsilon) * (reward + self.epsilon) ** self.alpha
        return priority

    def _get_transition_complexity(self, state1, state2):
        # Calculate the Euclidean distance as a measure of transition complexity
        complexity = np.linalg.norm(np.array(state1) - np.array(state2))
        return complexity

    def calculate_tcrp_priority(self, state1, state2, reward):
        # Calculate transition complexity
        complexity = self._get_transition_complexity(state1, state2)

        # Calculate and return TCRP priority
        priority = self._get_priority(complexity, reward)
        return priority

    # def _get_reward_factor(self, reward):
    #     reward_factor = (reward + 1e-5) ** (self.alpha * 0.5)  # Scale reward influence
    #     return reward_factor

    # def add(self, error, reward, sample):
    #     # Prioritize based on a combination of TD-error and reward
    #     priority = self._get_priority(error)
    #     reward_factor = self._get_reward_factor(reward)
    #     final_priority = priority * reward_factor
    #
    #     self.tree.add(final_priority, sample)

    # def add(self, state1, state2, reward, sample):
    #     # Calculate transition complexity
    #     complexity = self._get_transition_complexity(state1, state2)
    #
    #     # Calculate the final priority using TCRP
    #     priority = self._get_priority(complexity, reward)
    #
    #     # Add the sample with the calculated priority
    #     self.tree.add(priority, sample)

    def add(self, tcrp_value, reward, sample):
        # Use the combined priority for each experience
        priority = self._get_priority(tcrp_value, reward)
        self.tree.add(priority, sample)

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

    # def update(self, idx, error, reward):
    #     priority = self._get_priority(error)
    #     reward_factor = self._get_reward_factor(reward)
    #     final_priority = priority * reward_factor
    #     self.tree.update(idx, final_priority)

    def soft_update(self, idx, state1, state2, reward):
        # Recalculate priority when updating
        tcrp_value = self.calculate_tcrp_priority(state1, state2, reward)
        priority = self._get_priority(tcrp_value, reward)
        self.tree.update(idx, priority)

    # def soft_update(self, idx, tcrp_value, reward):
    #     # Update priority when TD-error or reward changes
    #     priority = self._get_priority(tcrp_value, reward)
    #     self.tree.update(idx, priority)


    def random_delete(self):
        total_priority = int(self.tree.total())  # calculate the total number of memories
        random_priority = random.uniform(0, total_priority)  # generates a random number from 1 to total-1

        idx, priority, data = self.tree.get(random_priority)  # searches through the sumtree and retrieves
                                                              # the random indexed memory

        self.tree.update(idx, 0)  # update the priority of the random memory to 0
                                         # so it will be deleted automatically


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity  # Set the capacity of the SumTree
        self.tree = np.zeros(2 * capacity - 1)  # Initialize the tree with zeros
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
