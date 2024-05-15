import torch
from pong_wrapper import PongWrapper
from dqn_model import DQN

def run_pong():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = PongWrapper(device=device)
    model = DQN(action_size=env.action_space.n).to(device)

    # Load a pretrained model if available
    # model.load_state_dict(torch.load('path_to_pretrained_model.pth'))

    state = env.reset()
    done = False
    while not done:
        action = model(state).argmax().item()  # Assuming the model predicts the next action
        state, reward, done, _ = env.step(action)

        # Optionally render the environment if you want to watch the game
        # env.render()

    print("Finished running Pong.")

if __name__ == '__main__':
    run_pong()

# Pong
# Space Invaders
# Asteroids
# Seaquest
# Frostbite
# Pac-Man
# Q*bert
# River Raid