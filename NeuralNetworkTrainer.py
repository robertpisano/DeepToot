import keras
import numpy as np
import pandas as pd
from keras.model import Sequential
from keras.layers import Dense, Dropout, LSTM

class LSTM_Model():
    def __init__():
        self.model = Sequential()
        self.model.add(LSTM(units = 50, input_shape = (50,15), return_sequences=True))
        self.model.add(LSTM())
        self.model.add(Dense(1))

if __name__ == __main__:
    None
