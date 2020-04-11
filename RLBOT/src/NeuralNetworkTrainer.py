import keras
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Lambda, TimeDistributed, Activation

class LSTM_Model():
    def __init__(self, training_data):
        self.model = Sequential()
        # self.model.add(Lambda(lambda x: np.expand_dims(x, axis=-1), input_shape=[None]))
        self.model.add(LSTM(units = 55, input_shape = (10, 55), return_sequences=True))
        self.model.add(LSTM(units = 55, return_sequences=True))
        # self.model.add(TimeDistributed(Dense(55)))
        # self.model.add(Activation('linear'))
        self.model.add(Lambda(lambda x: x[:, -5:, :]))
        self.model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-5, momentum=0.9),metrics=["mse"])

    def train(self, indata, outdata):
        history = self.model.fit(indata, outdata, epochs=50)     
        print(history)
if __name__ == "__main__":
    nn = LSTM_Model()
    # Print layer in/out shapes
    for layer in nn.model.layers:
        print(layer.input_shape, layer.output_shape)
    import dill
    datain = dill.load(open(r'D:\\Documents\\RL Replays\\rawin.p', 'rb'))
    dataout = dill.load(open(r'D:\\Documents\\RL Replays\\rawout.p', 'rb'))
    # print('datain ')
    # print(datain.shape)
    # print(' dataout ')
    # print(dataout.shape )
    nn.train(datain, dataout)