import gym
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

class DQNQbert(gym.Wrapper):
    def __init__(self, render_mode='rgb_array', device='cpu'):
        env = gym.make("ALE/Qbert-v5", render_mode=render_mode)
        super(DQNQbert, self).__init__(env)
        self.device = device
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((84, 84)),
            T.Grayscale(),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])

    def step(self, action):
        results = self.env.step(action)
        observation = results[0]  # Assume first item is the observation
        if isinstance(observation, tuple):  # Check if it's a tuple
            observation = observation[0]  # Use the first item as the actual observation
        processed_observation = self.process_observation(observation)
        return processed_observation, results[1], results[2], results[3]

    def reset(self):
        observation = self.env.reset()
        if isinstance(observation, tuple):  # Similar check as in step
            observation = observation[0]  # Use the first item as the actual observation
        return self.process_observation(observation)

    def process_observation(self, observation):
        if not isinstance(observation, (np.ndarray, torch.Tensor)):
            raise TypeError(f"Expected observation to be np.ndarray or Tensor, got {type(observation)}")
        observation = self.transform(observation).to(self.device)
        return observation.unsqueeze(0)  # Add batch dimension
