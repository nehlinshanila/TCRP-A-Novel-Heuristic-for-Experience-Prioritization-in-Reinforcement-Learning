
import torch
import os
from breakout import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

environment = DQNBreakout(device=device, rendor_mode='human')
state = environment.reset()

for _ in range(100):
    action = environment.action_space.sample()
    state, reward, done, info = environment.step(action)
