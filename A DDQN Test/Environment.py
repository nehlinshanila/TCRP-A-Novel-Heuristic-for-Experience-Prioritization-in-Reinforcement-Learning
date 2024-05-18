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

            r = np.clip(r, -1, 1)

            if done:  # terminal state
                s_ = None

            steps = agent.observe((s, a, r, s_))
            # steps = agent.observe(sample)
            agent.replay()

            s = s_
            R += r

            if done:
                break

        print(f'Total steps: {steps}   |    Total reward: {R}')
