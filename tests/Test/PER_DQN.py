from Huber_loss import huber_loss
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import tensorflow as tf


LR = 0.0003


class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self._build_model(state_size, action_size)

    def _build_model(self, state_size, action_size):
        model = Sequential()

        # model.add(Dense(24, input_dim=state_size, activation='relu'))
        # model.add(Dense(24, activation='relu'))
        # model.add(Dense(action_size, activation='linear'))

        model.add(Dense(units=64, activation='relu', input_dim=self.state_size))
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=self.action_size, activation='linear'))

        optimizer = Adam(learning_rate=LR)
        # model.compile(loss='mse', optimizer=Adam(learning_rate=LR))
        # loss = 'mse'
        loss = huber_loss

        model.compile(loss=loss, optimizer=optimizer)

        return model





