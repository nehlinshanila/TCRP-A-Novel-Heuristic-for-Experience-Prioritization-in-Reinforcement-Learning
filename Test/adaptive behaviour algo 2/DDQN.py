from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.optimizers import Adam
from sumTree import *


class DDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.dqn_learning_rate = 0.001

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        self.memory = Memory(1000000)  # PER Memory

        self.tau = 0.1  # this is a soft update rate for the target model
        self.batch_size = 32

        self.reward_threshold = 5.0  # Threshold for high rewards
        self.action_rewards = {a: [] for a in range(self.action_size)}  # Store rewards for each action

    def set_reward_threshold(self, reward_threshold):
        self.reward_threshold = reward_threshold

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=self.dqn_learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def soft_update_target_model(self):
        target_weights = self.target_model.get_weights()
        main_weights = self.model.get_weights()

        new_weights = []
        for main_weight, target_weight in zip(main_weights, target_weights):
            updated_w = self.tau * main_weight + (1 - self.tau) * target_weight
            new_weights.append(updated_w)

        self.target_model.set_weights(new_weights)

    def memorize(self, state, action, reward, next_state, done):

        target = self.model.predict(state)

        if done:
            target[0][action] = reward
        else:
            best_next_action = np.argmax(self.model.predict(next_state)[0])
            target[0][action] = reward + self.gamma * self.target_model.predict(next_state)[0][best_next_action]

        # Compute TD-error
        current_q_value = self.model.predict(state)[0][action]
        error = abs(target[0][action] - current_q_value)

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
        exp_act_values = np.exp(act_values[0] / self.tau)
        boltzman_probabilities = exp_act_values / np.sum(exp_act_values)

        probabilities = boltzman_probabilities.copy()

        if high_reward_actions:
            for a in high_reward_actions:
                probabilities[a] += 0.1  # Increase the probability for high-reward actions

            probabilities /= probabilities.sum()

        return np.random.choice(range(self.action_size), p=probabilities)

    def replay(self, batch_size=32):
        """......................................................................."""
        # if len(self.memory) < batch_size:
        #     return  # Don't replay until there's enough samples

        # Sample a batch of experiences from memory
        minibatch, idxs, is_weights = self.memory.sample(batch_size)

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                # Double DQN: Use main model to select action, target model to evaluate it
                best_next_action = np.argmax(self.model.predict(next_state)[0])
                target[0][action] = reward + self.gamma * self.target_model.predict(next_state)[0][best_next_action]

                # next_q_values = self.model.predict(next_state)
                # target[0][action] = reward + self.gamma * np.amax(next_q_values[0])

            # Compute TD-error
            current_q_value = self.model.predict(state)[0][action]
            error = abs(target[0][action] - current_q_value)

            # Update the memory with the new priority that incorporates reward
            self.memory.update(idxs[i], error, reward)

            self.model.train_on_batch(state, target)

        # Soft update the target model after every replay step
        self.soft_update_target_model()

        # Reduce epsilon to encourage exploitation over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """ Load pre-trained model """
        self.model.load_weights(name)

    def save(self, name):
        """ Save trained model """
        self.model.save_weights(name)
