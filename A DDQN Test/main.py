from Environment import Environment
from agent import Agent, RandomAgent, MEMORY_CAPACITY, RANDOM_AGENT_CAPACITY

if __name__ == "__main__":

    PROBLEM = 'CartPole-v1'

    env = Environment(PROBLEM)

    state_size = env.env.observation_space.shape[0]
    action_size = env.env.action_space.n

    agent = Agent(state_size, action_size)
    randomAgent = RandomAgent(action_size)

    try:
        print("Initialization with random agent...")
        while randomAgent.exp < RANDOM_AGENT_CAPACITY:
            env.run(randomAgent)
            print(randomAgent.exp, "/", RANDOM_AGENT_CAPACITY)

        agent.memory = randomAgent.memory

        randomAgent = None

        print("Starting learning")
        while True:
            env.run(agent)
    finally:
        agent.brain.model.save("DDQN-PER.keras")
