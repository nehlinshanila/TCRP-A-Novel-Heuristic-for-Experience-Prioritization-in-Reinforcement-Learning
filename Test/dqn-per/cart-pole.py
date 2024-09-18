from dqn import *
import gym
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
import datetime
import tensorflow as tf


# Create TensorBoard callback
log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# alada logs/dqn or Log/ddqn
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Lists to store metrics for each episode
episode_rewards = []
episode_epsilons = []

# to write the tensorboard logs
writer = tf.summary.create_file_writer(log_dir)


env = gym.make('CartPole-v1', render_mode='human')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
scores = []
EPISODES = 2000
avg_window = 100

for e in range(EPISODES):
    state, _ = env.reset(seed=42)
    # print(f'state : {state}')
    state = np.reshape(state, [1, state_size])
    done = False
    time = 0
    total_reward = 0

    while not done:
        time += 1
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        agent.memorize(state, action, reward, next_state, done)
        state = next_state

        total_reward += reward

        if done:
            scores.append(total_reward)
            if e % 100 == 0 and e > 1:
                print("episode: {}/{}, Score Mean: {} / Median: {} ".format(e, EPISODES, int(np.mean(scores)), int(np.median(scores))))
                print("Beta {:.5f} / Eps: {:.5f}".format(agent.memory.beta, agent.epsilon))
            # scores.append(time)
    if agent.memory.tree.n_entries > 1000:
        agent.replay()

    with writer.as_default():
        tf.summary.scalar('Total Reward', total_reward, step=e)
        tf.summary.scalar('Epsilon', agent.epsilon, step=e)

        # Compute and log the average reward over the last 100 episodes
        if len(scores) >= avg_window:
            avg_reward = np.mean(scores[-avg_window:])
            tf.summary.scalar('Average Reward (last 100 episodes)', avg_reward, step=e)

env.close()


# https://github.com/jcborges/dqn-per/tree/master
