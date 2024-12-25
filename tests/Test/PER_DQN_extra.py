import random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
LR = 0.001
TAU = 0.005
MEMORY_SIZE = 10000
ALPHA = 0.6
BETA_START = 0.4
BETA_INCREMENT_PER_STEP = 0.001
UPDATE_EVERY = 4

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        print(f'the max priorites: {self.priorities}')
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)

    def sample(self, batch_size, beta):
        if len(self.buffer) == self.capacity:
            priorities = np.array(self.priorities)
        else:
            priorities = np.array(self.priorities)[:len(self.buffer)]

        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
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
            self.priorities[idx] = priority


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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory.buffer) > BATCH_SIZE:
                experiences = self.memory.sample(BATCH_SIZE, self.beta)
                self.learn(experiences)
                self.beta = min(1.0, self.beta + self.beta_increment_per_step)

    def act(self, state):
        # Ensure state is extracted correctly
        if isinstance(state, tuple):
            state = state[0]

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if np.random.rand() <= self.epsilon:
            return random.choice(range(self.action_size))
        else:
            with torch.no_grad():
                action_values = self.policy_net(state)
            return np.argmax(action_values.cpu().data.numpy())

    def learn(self, experiences):
        states, actions, rewards, next_states, dones, indices, weights = experiences

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        Q_expected = self.policy_net(states).gather(1, actions)
        Q_targets_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        loss = (Q_expected - Q_targets).pow(2) * weights
        prios = loss + 1e-5
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.memory.update_priorities(indices, prios.detach().cpu().numpy())

        self.soft_update(self.policy_net, self.target_net, TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, filename):
        torch.save(self.policy_net.state_dict(), filename)

    def load(self, filename):
        self.policy_net.load_state_dict(torch.load(filename))
        self.policy_net.eval()


agent = Agent(state_size, action_size)

num_episodes = 1000
max_t = 1000

for e in range(num_episodes):
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    total_reward = 0
    for t in range(max_t):
        action = agent.act(state)
        next_state, reward, done, truncated, info = env.step(action)
        if isinstance(next_state, tuple):
            next_state = next_state[0]
        agent.step(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done or truncated:
            break
    agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
    print(f"Episode {e}/{num_episodes} - Total Reward: {total_reward} - Epsilon: {agent.epsilon}")

env.close()
