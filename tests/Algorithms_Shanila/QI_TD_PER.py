import sys
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from collections import deque

# Adjust the path to ensure the `Atari.ATARI.Pong` module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Atari/ATARI/Pong')))

from Atari.ATARI.Pong.pong import Pong

# Set the device to use for computations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(84 * 84, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Define the experience replay buffer
class QI_TD_PER_Buffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.alpha = alpha

    def add(self, experience, td_error, temporal_score, diversity_score):
        max_priority = self.priorities.max() if self.buffer else 1.0
        priority = self.calculate_priority(td_error, temporal_score, diversity_score)
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        self.priorities[self.pos] = max_priority if priority is None else priority
        self.pos = (self.pos + 1) % self.capacity

    def calculate_priority(self, td_error, temporal_score, diversity_score):
        td_error = max(td_error, 0.0001)
        combined_priority = td_error ** self.alpha + temporal_score + diversity_score
        return combined_priority

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]

        priorities = np.nan_to_num(priorities, nan=0.0001)

        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, weights

    def update_priorities(self, indices, td_errors, temporal_scores, diversity_scores):
        for idx, (td_error, temporal_score, diversity_score) in zip(indices,
                                                                    zip(td_errors, temporal_scores, diversity_scores)):
            self.priorities[idx] = self.calculate_priority(td_error, temporal_score, diversity_score)

# Hyperparameters
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995
batch_size = 32
learning_rate = 0.001
target_update = 10
memory_capacity = 10000

# Initialize environment, Q-network, and replay buffer
env = Pong(render_mode='human', device=device)
state_size = 84 * 84
action_size = env.action_space.n
q_network = QNetwork(action_size).to(device)
target_network = QNetwork(action_size).to(device)
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
memory = QI_TD_PER_Buffer(memory_capacity)

def preprocess_state(state):
    return state.view(-1).unsqueeze(0).to(device)

def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            q_values = q_network(state)
            return q_values.argmax().item()

def train_step():
    if len(memory.buffer) < batch_size:
        return

    experiences, indices, weights = memory.sample(batch_size)
    weights = torch.tensor(weights, device=device, dtype=torch.float)

    states, actions, rewards, next_states, dones = zip(*experiences)

    states = torch.cat(states)
    actions = torch.tensor(actions, device=device).unsqueeze(1)
    rewards = torch.tensor(rewards, device=device)
    next_states = torch.cat(next_states)
    dones = torch.tensor(dones, device=device, dtype=torch.float)

    q_values = q_network(states).gather(1, actions)
    next_q_values = target_network(next_states).max(1)[0].detach()
    target = rewards + gamma * next_q_values * (1 - dones)

    loss = (weights * (q_values.squeeze() - target) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    td_errors = (q_values.squeeze() - target).detach().cpu().numpy()
    temporal_scores = np.random.random(batch_size)
    diversity_scores = np.random.random(batch_size)
    memory.update_priorities(indices, td_errors, temporal_scores, diversity_scores)

def main():
    global epsilon
    num_episodes = 500
    for e in range(num_episodes):
        state = preprocess_state(env.reset())
        total_reward = 0
        done = False

        while not done:
            action = select_action(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            next_state = preprocess_state(next_state)
            memory.add((state, action, reward, next_state, done), reward, 1.0, 1.0)
            state = next_state
            total_reward += reward

            train_step()

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if e % target_update == 0:
            target_network.load_state_dict(q_network.state_dict())

        logger.info(f"Episode {e + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")
        print(f"Episode {e + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}", flush=True)

if __name__ == '__main__':
    main()
