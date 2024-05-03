import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from Huber_loss import huber_loss
import random
from collections import deque
from Memory import Memory


class DDQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0003

        self.model = self._build_model()
        self.model_ = self._build_model()  # target model

    def _build_model(self, model):
        # this is the neural net for DQN
        model = Sequential()

        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=self.state_size,
                         data_format='channels_first'))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(units=512, activation='relu'))

        model.add(Dense(units=self.action_size, activation='linear'))

        opt = RMSprop(lr=self.learning_rate)
        model.compile(loss=huber_loss, optimizer=opt)

        return model

    def train(self, x, y, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size=32, epochs=epochs, verbose=verbose)

    def predict(self, s, target=False):
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)

    def predictOne(self, s, target=False):
        return self.predict(s.reshape(1, self.state_size), target).flatten()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)

        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        target_model = self._build_model()
        target_model.set_weights(self.model.get_weights())

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                best_next_action = np.argmax(self.model.predict(next_state)[0])

                target = (reward + self.gamma * target_model.predict(next_state)[0][best_next_action])

            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def updateTargetModel(self):
        self.model_.set_weights(self.model.get_weights())
