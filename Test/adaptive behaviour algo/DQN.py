
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sumTree import *


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.dqn_learning_rate = 0.001
        self.model = self._build_model()

        self.memory = Memory(1000000)  # PER Memory

        self.reward_threshold = 5.0  # Threshold for high rewards
        self.action_rewards = {a: [] for a in range(self.action_size)}  # Store rewards for each action

    def set_reward_threshold(self, reward_threshold):
        self.reward_threshold = reward_threshold

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=self.dqn_learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):

        target = self.model.predict(state)
        if done:
            target[0][action] = reward
        else:
            next_q_values = self.model.predict(next_state)
            target[0][action] = reward + self.gamma * np.amax(next_q_values[0])

        # Compute TD-error
        error = abs(np.real(target[0][action] - self.model.predict(state)[0][action]))
        """...............keep adding the new experiences in memory..................."""
        self.memory.add(error, reward, (state, action, reward, next_state, done))

    def act(self, state):
        # Exploration: choose a random action
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            # Exploitation: predict the action based on the Q-values
            act_values = self.model.predict(state)

            # Reward-based action bias
            high_reward_actions = self.get_high_reward_actions(state)

            if high_reward_actions:
                # Add some probability to select actions with high past rewards
                action = self.bias_towards_high_reward_actions(act_values, high_reward_actions)
            else:
                action = np.argmax(act_values[0])

            return action

    def past_rewards_for_action(self, action):
        # Calculate the average reward for the given action
        if len(self.action_rewards[action]) == 0:
            return 0
        return np.mean(self.action_rewards[action])

    def get_high_reward_actions(self, state):
        # Identify actions that consistently lead to high rewards
        # This can be stored or computed based on experience
        high_reward_actions = []
        for a in range(self.action_size):
            avg_reward = self.past_rewards_for_action(a)
            if avg_reward > self.reward_threshold:
                high_reward_actions.append(a)
        return high_reward_actions

    def bias_towards_high_reward_actions(self, act_values, high_reward_actions):
        # Introduce a bias to select one of the high-reward actions
        probabilities = np.ones(self.action_size) * 0.1  # Small probability for each action
        for a in high_reward_actions:
            probabilities[a] += 0.2  # Increase probability for high-reward actions
        probabilities /= probabilities.sum()  # Normalize
        return np.random.choice(range(self.action_size), p=probabilities)

    def replay(self, batch_size=32):
        """......................................................................."""

        # Sample a batch of experiences from memory
        minibatch, idxs, is_weights = self.memory.sample(batch_size)

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                next_q_values = self.model.predict(next_state)
                target[0][action] = reward + self.gamma * np.amax(next_q_values[0])

            # Compute TD-error
            error = abs(target[0][action] - self.model.predict(state)[0][action])

            # Update the memory with the new priority that incorporates reward
            self.memory.update(idxs[i], error, reward)
            self.model.train_on_batch(state, target)

        # Reduce epsilon to encourage exploitation over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
