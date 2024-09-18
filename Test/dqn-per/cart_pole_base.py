import gymnasium as gym
from stable_baselines3 import DQN
import datetime

log_dir = "logs/base" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

env = gym.make('CartPole-v1', render_mode='human')
model = DQN("MlpPolicy", env, verbose=1, seed=42, tensorboard_log=log_dir)

model.learn(total_timesteps=10000, log_interval=4)

env.close()
