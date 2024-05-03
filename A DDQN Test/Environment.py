import gym
import numpy as np


class Environment:
    def __init__(self, problem):
        self.problem = problem
        self.env = gym.make(problem, render_mode='human')

    def run(self, agent):

        s = self.env.reset()

        R = 0
        while True:
            a = agent.act(s)

            r = 0

            s_, r, done, truncated, info = self.env.step(a)
            # s_ = np.array([s[1], processImage(img)])  # last two screens

            r = np.clip(r, -1, 1)  # clip reward to [-1, 1]

            if done:  # terminal state
                s_ = None

            agent.observe((s, a, r, s_))
            agent.replay()

            s = s_
            R += r

            if done:
                break

        print("Total reward:", R)
