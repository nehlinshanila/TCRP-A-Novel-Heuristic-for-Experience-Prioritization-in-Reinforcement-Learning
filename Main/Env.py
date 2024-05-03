# import gymnasium as gym
import numpy as np
import gym

env = gym.make("ALE/Breakout-v5", render_mode="human")

# gymnasium.make("ALE/Breakout-v5")
obs, info = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, t, tr, i = env.step(action)

    if t or tr:
        obs, i = env.reset()

env.close()


# class Environment:
#     def __init__(self, problem):
#         self.problem = problem
#         self.env = gym.make(problem, render_mode="human")
#
#     def run(self, agent):
#         obs = self.env.reset()
#         w = processImage(img)
#         s = np.array([w, w])
#
#         R = 0
#         while True:
#             # self.env.render()
#             a = agent.act(s)
#
#             r = 0
#             img, r, done, info = self.env.step(a)
#             s_ = np.array([s[1], processImage(img)])  # last two screens
#
#             r = np.clip(r, -1, 1)  # clip reward to [-1, 1]
#
#             if done:  # terminal state
#                 s_ = None
#
#             agent.observe((s, a, r, s_))
#             agent.replay()
#
#             s = s_
#             R += r
#
#             if done:
#                 break
#
#         print("Total reward:", R)
#
