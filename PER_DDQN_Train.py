import torch
import torch.optim as optim
import numpy as np
from PER import PER
from PER_DDQN import DDQN
from Main.Set_Device import device

gamma = 0.99
epsilon = 0.9
batch_size = 32
learning_rate = 1e-3
alpha = 0.6


# input and output dim must be set dynamically somehow
memory = PER(capacity=10000, alpha=alpha)
policy_net = DDQN(input_dim=input_dim, output_dim=output_dim).to(device)
target_net = DDQN(input_dim=input_dim, output_dim=output_dim).to(device)

target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)


def update_model():
    # sample a batch with priorities
    batch, idxs, is_weights = memory.sample(batch_size)

    # Convert to tensors, move to device
    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.bool).to(device)


    current_q_value = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Calculate TD-Target
    # (Get next_state Q-values from both current and target networks for DDQN)
    with torch.no_grad():
        next_states_q_values = policy_net(next_states)
        best_next_actions = next_states_q_values.max(1)[1].unsqueeze(1)  # DDQN: Select actions according to policy_net

        next_states_target_q_values = target_net(next_states).gather(1, best_next_actions)  # DDQN: Evaluate with target_net

        td_target = rewards + (gamma * next_states_target_q_values * (1 - dones))


    # calculate the loss using importance sampling weights
    loss = (td_target - current_q_value) ** 2 * torch.tensor(is_weights).to(device)
    loss = loss.mean()

    # optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # update the priorities in sum-tree
    new_priorities = np.power(np.abs(td_target - current_q_value) + epsilon, alpha).detach().cpu().numpy()

    # new_priorities =
    memory.update_priorities(idxs, new_priorities)
