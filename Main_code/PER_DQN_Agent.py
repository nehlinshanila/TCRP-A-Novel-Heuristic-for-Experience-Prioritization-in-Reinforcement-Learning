from Main_code.Priortized_Replay_Buffer import PrioritizedReplayBuffer, np, random
from image_process import preprocess_state

from PER_DQN import DQN

# Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

TAU = 0.005
MEMORY_SIZE = 10000
ALPHA = 0.6
BETA_START = 0.4
BETA_INCREMENT_PER_STEP = 0.001
UPDATE_EVERY = 4


class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = PrioritizedReplayBuffer(MEMORY_SIZE, ALPHA)
        self.gamma = GAMMA
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_END
        self.epsilon_decay = EPSILON_DECAY
        self.beta = BETA_START
        self.beta_increment_per_step = BETA_INCREMENT_PER_STEP
        self.update_every = UPDATE_EVERY

        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.update_target_model()

        self.t_step = 0

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(range(self.action_size))
        q_values = self.policy_net.model.predict(state)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        state = np.reshape(state, [1, self.state_size])
        next_state = np.reshape(next_state, [1, self.state_size])
        self.memory.add(1.0, (state, action, reward, next_state, done))

    def learn(self, experiences):
        states, actions, rewards, next_states, dones, indices, weights = experiences

        q_targets = self.target_net.model.predict_on_batch(next_states)
        q_targets_next = np.amax(q_targets, axis=1)

        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

        q_expected = self.policy_net.model.predict_on_batch(states)

        for i, action in enumerate(actions):
            q_expected[i][action] = q_targets[i]

        loss = self.policy_net.model.train_on_batch(states, q_expected, sample_weight=weights)

        td_errors = q_targets - np.amax(q_expected, axis=1)
        self.memory.update_priorities(indices, np.abs(td_errors) + 1e-5)

        return loss

    def update_target_model(self):
        self.target_net.model.set_weights(self.policy_net.model.get_weights())

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        experiences = self.memory.sample(batch_size, self.beta)
        return self.learn(experiences)

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory.tree.data) > BATCH_SIZE:
                experiences = self.memory.sample(BATCH_SIZE, self.beta)
                self.learn(experiences)
                self.beta = min(1.0, self.beta + self.beta_increment_per_step)

    def save(self, filename):
        self.policy_net.model.save(filename)

    def load(self, filename):
        self.policy_net.model.load_weights(filename)
        self.update_target_model()

    def predict(self, state):
        return self.policy_net.model.predict(state)[0]

    def evaluate(self, env, num_episodes=10):
        total_rewards = []
        for _ in range(num_episodes):
            state = env.reset()

            # if isinstance(state, tuple):
            #     state = state[0]

            state = preprocess_state(state)

            state = np.reshape(state, [1, self.state_size])
            total_reward = 0
            done = False
            while not done:
                action = np.argmax(self.predict(state))
                next_state, reward, done, truncated, _ = env.step(action)
                next_state = preprocess_state(next_state)
                next_state = np.reshape(next_state, [1, self.state_size])
                total_reward += reward
                state = next_state
                if done or truncated:
                    break
            total_rewards.append(total_reward)
        return np.mean(total_rewards), np.std(total_rewards)

