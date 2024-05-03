import torch
import torch.nn as nn
import random
from Main.Set_Device import device


class DDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Linear(128, self.output_dim)
        )

    def forward(self, state):
        QValues = self.layers(state)
        return QValues

    def actor(self, state, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                q_values = self(state)
                action = q_values.max(1)[1].item()  # this is basically the greedy action

        else:
            action = random.randrange(self.output_dim)  # if the epsilon value not higher, then take random actions

        return action
