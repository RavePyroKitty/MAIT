import keras
import numpy as np
import tensorflow as tf
from keras.layers import MaxPooling2D, ConvLSTM2D, Embedding, Dense

from sentimental_analysis.sentimental_features import SentimentalFeatures


# TODO: Organize this code into a class that fits with the rest of the syntax of the program

class SentimentalThinker(SentimentalFeatures):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embedded_sentences = np.array(self.textual_features())

    def format_data(self):
        print(self.embedded_sentences.shape)

        return self.embedded_sentences.shape

    def think(self):
        inputs = self.format_data()

        tf.random.set_seed(125)
        randomized = tf.random.uniform(inputs)
        print(randomized)

        model = keras.Sequential()
        model.add(ConvLSTM2D(kernel_size=3, filters=64, activation='tanh'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(ConvLSTM2D(kernel_size=3, filters=64, activation='softmax'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Embedding(output_dim=64))
        model.add(Dense(units=200, activation='softmax'))
        model.add(Dense(units=3, activation='tanh'))
        model.compile(loss='rmse', optimizer='adam')
        model.summary()
