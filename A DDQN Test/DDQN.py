from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from Huber_loss import huber_loss
from keras.callbacks import TensorBoard

# Initialization
log_dir = "./logs"
tensorboard_callback = TensorBoard(log_dir=log_dir)


class Brain:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.0003

        self.model = self._createModel()
        self.model_ = self._createModel()  # target network

    def _createModel(self):
        model = Sequential()

        model.add(Dense(units=64, activation='relu', input_dim=self.state_size))
        model.add(Dense(units=128, activation='relu'))
        model.add(Dense(units=128, activation='relu'))
        model.add(Dense(units=self.action_size, activation='linear'))

        opt = RMSprop(learning_rate=self.learning_rate)
        model.compile(loss=huber_loss, optimizer=opt)

        return model

    def train(self, x, y, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size=64, epochs=epochs, verbose=verbose, callbacks=[tensorboard_callback])

    def predict(self, s, target=False):
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)

    def predictOne(self, s, target=False):
        return self.predict(s.reshape(1, self.state_size), target=target).flatten()

    def updateTargetModel(self):
        self.model_.set_weights(self.model.get_weights())
