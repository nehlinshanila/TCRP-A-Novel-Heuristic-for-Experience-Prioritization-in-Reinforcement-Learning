import torch
import numpy as np
from collections import deque
import logging
import sys
import os
import random
import torch.nn as nn
import torch.optim as optim

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
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class HD_GOER_Buffer:
    def __init__(self, capacity, dim=10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.dim = dim
        self.goal_vector = self.generate_random_hypervector()

    def generate_random_hypervector(self):
        return np.random.choice([-1, 1], size=(self.dim,))

    def add(self, experience, td_error, goal):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        self.priorities[self.pos] = max(abs(td_error), 0.0001)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None, None, None
        priorities = np.clip(self.priorities[:len(self.buffer)], 1e-6, None)
        probabilities = priorities ** 0.6
        probabilities /= probabilities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        weights = (1 / (len(self.buffer) * probabilities[indices])) ** 0.4
        weights /= weights.max()
        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = max(abs(error), 0.0001)

# Hyperparameters
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995
batch_size = 32
learning_rate = 0.001
target_update = 10
memory_capacity = 10000
hypervector_dim = 10000

# Initialize environment, Q-network, and replay buffer
env = Pong(render_mode='human', device=device)
action_size = env.action_space.n
q_network = QNetwork(action_size).to(device)
target_network = QNetwork(action_size).to(device)
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
memory = HD_GOER_Buffer(memory_capacity, dim=hypervector_dim)

def preprocess_state(state):
    if not isinstance(state, torch.Tensor):
        state = torch.from_numpy(state)
    return state.float().view(-1).unsqueeze(0).to(device)

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
    if experiences is None:
        return
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
    memory.update_priorities(indices, td_errors)

def main():
    global epsilon
    num_episodes = 500
    for e in range(num_episodes):
        state = preprocess_state(env.reset())
        total_reward = 0
        done = False

        while not done:
            action = select_action(state, epsilon)
            # Use extended unpacking to handle additional returned values
            result = env.step(action)
            next_state, reward, done, *extra = result
            next_state = preprocess_state(next_state)
            memory.add((state, action, reward, next_state, done), reward, memory.goal_vector)
            state = next_state
            total_reward += reward
            train_step()

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if e % target_update == 0:
            target_network.load_state_dict(q_network.state_dict())

        logger.info(f"Episode {e + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

if __name__ == '__main__':
    main()