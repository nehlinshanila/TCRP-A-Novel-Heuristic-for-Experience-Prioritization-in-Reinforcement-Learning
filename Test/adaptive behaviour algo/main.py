import gymnasium as gym
from DQN import *

from tensorflow.keras.callbacks import TensorBoard
import datetime
import tensorflow as tf


if __name__ == "__main__":
    """..............................for tensorboard logs..................................."""
    log_dir = "logs/AdaptiveBehavior" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    episode_rewards = []
    episode_epsilons = []

    writer = tf.summary.create_file_writer(log_dir)

    """..................,,,,,,,.the main training loop....................................."""
    env = gym.make('CartPole-v1', render_mode='human')

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)
    agent.set_reward_threshold(250)

    scores = []
    EPISODES = 2000
    batch_size = 32
    avg_window = 100

    for e in range(EPISODES):
        state, _ = env.reset(seed=42)
        state = np.reshape(state, [1, state_size])
        done = False
        total_reward = 0
        time = 0

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
                    print("episode: {}/{}, Score Mean: {} / Median: {} ".format(e, EPISODES, int(np.mean(scores)),
                                                                                int(np.median(scores))))
                    print("Beta {:.5f} / Eps: {:.5f}".format(agent.memory.beta, agent.epsilon))
                break

        if agent.memory.tree.n_entries > 1000:
            agent.replay(batch_size)

        with writer.as_default():
            tf.summary.scalar('Total Reward', total_reward, step=e)
            tf.summary.scalar('Epsilon', agent.epsilon, step=e)

            # Compute and log the average reward over the last 100 episodes
            if len(scores) >= avg_window:
                avg_reward = np.mean(scores[-avg_window:])
                tf.summary.scalar('Average Reward (last 100 episodes)', avg_reward, step=e)

    env.close()

