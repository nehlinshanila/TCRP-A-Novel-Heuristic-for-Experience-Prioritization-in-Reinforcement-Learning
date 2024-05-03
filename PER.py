from Main.SumTree import SumTree, np
import random


class PER:
    def __init__(self, capacity, alpha, beta, beta_increment):
        self.buffer = []
        self.priorities = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0

    def add(self, experience):
        self.buffer.append(experience)
        self.priorities.add(self.max_priority, experience)

    def sample(self, batch_size):
        batch = []
        idxs = []
        segment = self.priorities.total() / batch_size
        is_weights = np.zeros((batch_size, 1))

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, priority, data) = self.priorities.get(s)
            is_weights[i, 0] = (self.priorities.total() * priority) ** (-self.beta)
            batch.append(data)
            idxs.append(idx)

        is_weights /= is_weights.max()
        return batch, idx, is_weights

    def update_priorities(self, idxs, priorities):
        for idx, priority in zip(idxs, priorities):
            self.priorities._propagate(idx, priority - self.priorities[idx])  # update the priority
            self.max_priority = max(self.max_priority, priority)

        self.beta = min(1.0, self.beta + self.beta_increment)
