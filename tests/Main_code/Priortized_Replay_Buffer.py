from SumTree import SumTree, np
import random


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.tree = SumTree(capacity)
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001
        self.epsilon = 0.01
        self.position = 0

    def add(self, error, sample):
        # max_priority = np.max(self.tree.tree[-self.capacity:]) if self.position else 1.0
        # self.tree.add(max_priority, (state, action, reward, next_state, done))
        p = (error + self.epsilon) ** self.alpha
        self.tree.add(p, sample)

    def sample(self, batch_size, beta):
        samples = []
        indices = []
        segment = self.tree.total() / batch_size

        priorities = []

        # self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

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

    # def update_priorities(self, batch_indices, batch_priorities):
    #     for idx, priority in zip(batch_indices, batch_priorities):
    #         self.tree.update(idx, priority)

    def update_priorities(self, idxs, errors):
        for idx, error in zip(idxs, errors):
            p = (error + self.epsilon) ** self.alpha
            self.tree.update(idx, p)

    def __len__(self):
        return len(self.tree.data) - np.count_nonzero(self.tree.data == 0)

