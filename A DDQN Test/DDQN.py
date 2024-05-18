from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from Huber_loss import huber_loss
from keras.callbacks import TensorBoard


class Brain:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.0003

        self.model = self._build_model()
        self.target_model = self._build_model()  # target network

        # Initialization
        log_dir = "./logs"
        self.tensorboard_callback = TensorBoard(log_dir=log_dir)

    def _build_model(self):
        model = Sequential()

        model.add(Dense(units=64, activation='relu', input_dim=self.state_size))
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=self.action_size, activation='linear'))

        optimizer = RMSprop(learning_rate=self.learning_rate)
        model.compile(loss=huber_loss, optimizer=optimizer)

        return model

    def train(self, x, y, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size=64, epochs=epochs, verbose=verbose, callbacks=self.tensorboard_callback)

    def predict(self, s, target=False):
        if target:
            print('target state in predict: ', s)
            return self.target_model.predict(s)
        else:
            print('state in predict: ', s.shape())
            return self.model.predict(s)

    def predictOne(self, s, target=False):
        return self.predict(s.reshape(1, self.state_size), target=target).flatten()

    def updateTargetModel(self):
        self.target_model.set_weights(self.model.get_weights())
