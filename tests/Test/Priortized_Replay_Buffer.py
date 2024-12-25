from SumTree import SumTree, np
import random


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.alpha = alpha
        self.tree = SumTree(capacity)
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        max_priority = np.max(self.tree.tree[-self.capacity:]) if self.position else 1.0
        self.tree.add(max_priority, (state, action, reward, next_state, done))

    def sample(self, batch_size, beta):
        indices = []
        priorities = []
        samples = []

        segment = self.tree.total() / batch_size
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            priorities.append(p)
            indices.append(idx)
            samples.append(data)

        sampling_probabilities = priorities / self.tree.total()
        weights = (len(self.tree.data) * sampling_probabilities) ** (-beta)
        weights /= weights.max()

        batch = list(zip(*samples))
        states = np.vstack(batch[0])
        actions = np.array(batch[1])
        rewards = np.array(batch[2])
        next_states = np.vstack(batch[3])
        dones = np.array(batch[4], dtype=np.uint8)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.tree.update(idx, priority)

    def __len__(self):
        return len(self.tree.data) - np.count_nonzero(self.tree.data == 0)

