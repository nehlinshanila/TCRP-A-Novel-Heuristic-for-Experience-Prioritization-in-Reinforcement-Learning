import gym
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

class DQNSpaceInvaders(gym.Wrapper):
    def __init__(self, render_mode='rgb_array', device='cpu'):
        env = gym.make("ALE/SpaceInvaders-v5", render_mode=render_mode)
        super(DQNSpaceInvaders, self).__init__(env)
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
        print("Step results:", results)  # Debugging output
        observation = results[0]
        if isinstance(observation, tuple):
            print("Unexpected tuple in step observation:", observation)  # More detailed debug output
            observation = observation[0]  # This line may need adjustment based on actual tuple structure
        processed_observation = self.process_observation(observation)
        return processed_observation, results[1], results[2], results[3]

    def reset(self):
        observation = self.env.reset()
        print("Reset observation:", observation)  # Debugging output
        if isinstance(observation, tuple):
            print("Unexpected tuple in reset observation:", observation)  # More detailed debug output
            observation = observation[0]  # This line may need adjustment based on actual tuple structure
        return self.process_observation(observation)

    def process_observation(self, observation):
        # Ensure observation is in correct format before processing
        if not isinstance(observation, (np.ndarray, torch.Tensor)):
            raise TypeError(f"Expected observation to be np.ndarray or Tensor, got {type(observation)}")
        observation = self.transform(observation).to(self.device)
        return observation.unsqueeze(0)  # Add batch dimension
