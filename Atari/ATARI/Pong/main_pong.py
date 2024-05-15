import torch
from pong import DQNPong

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    environment = DQNPong(render_mode='human', device=device)
    state = environment.reset()

    while True:
        action = environment.action_space.sample()  # Randomly sample an action
        next_state, reward, done, _ = environment.step(action)
        if done:
            state = environment.reset()  # Reset the environment if the game is over
        else:
            state = next_state

        print(f"Action taken: {action}, Reward: {reward}")

if __name__ == '__main__':
    main()
