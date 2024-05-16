import gym
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

class DQNBreakout(gym.Wrapper):
    def __init__(self, render_mode='rgb_array', device='cpu'):
        env = gym.make("ALE/Breakout-v5", render_mode=render_mode)
        super(DQNBreakout, self).__init__(env)
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
        observation = results[0]
        reward = results[1]
        done = results[2]
        info = results[3] if len(results) > 3 else {}  # Handle cases where more than 4 values might be returned
        processed_observation = self.process_observation(observation)
        return processed_observation, reward, done, info

    def reset(self):
        observation = self.env.reset()
        return self.process_observation(observation)

    def process_observation(self, observation):
        if isinstance(observation, tuple):
            observation = observation[0]  # Handling tuple output from environment
        if not isinstance(observation, (np.ndarray, torch.Tensor)):
            raise TypeError(f"Expected observation to be np.ndarray or Tensor, got {type(observation)}")
        observation = self.transform(observation).to(self.device)
        return observation.unsqueeze(0)  # Add batch dimension
