import gymnasium as gym

from ddqn import DDQN
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Flatten, ZeroPadding2D
from keras.optimizers import RMSprop
from keras import backend as K
# K.set_image_dim_ordering('tf')

# Use any environment here. OpenAI's Gym can be used
# Must ensure that model fits the dimensions of data returned by environment
env = gym.make('CartPole-v1', render_mode='human')

# cols, rows = env.shape()
frames = 4

# shape = (cols, rows, frames)  # specific to arbitrary environment
num_actions = env.action_space.n             # specific to arbitrary environment
state_size = env.observation_space.shape[0]

def create_model():
    model = Sequential()

    # Input Layer (size = 4 for CartPole)
    model.add(Dense(units=24, activation='relu', input_shape=(state_size,)))

    # Hidden Layer
    model.add(Dense(units=24, activation='relu'))

    # Output Layer (2 actions: left or right)
    model.add(Dense(units=num_actions, activation='linear'))

    # optimizer and loss
    lr = 0.001
    losss = "mse"
    rmsprop = RMSprop(learning_rate=lr, clipvalue=1.0)

    model.compile(loss=losss, optimizer=rmsprop)

    return rmsprop, losss, lr, model


optimizer, loss, learning_rate, model = create_model()

ddqn = DDQN(env, model, loss, optimizer, learning_rate, num_actions, 500, (state_size, 1, 1))

while True:
    s = env.reset()
    done = False

    while not done:
        env.render()  # Visualize the CartPole environment
        ddqn.act_and_learn()


    # show = env.display()
    # ddqn.act_and_learn()
    # Environment automatically resets
    # Check ddqn.py for functions which can print statistics here