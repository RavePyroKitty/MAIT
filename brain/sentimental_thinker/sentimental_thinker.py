import keras
import numpy as np
from keras.layers import MaxPooling2D, ConvLSTM2D, Embedding, Dense, Flatten, GlobalMaxPooling1D, Conv1D
from official.nlp.modeling.layers import RandomFeatureGaussianProcess
from tensorflow_addons.layers import SpectralNormalization

from sentimental_analysis.sentimental_features import SentimentalFeatures


# from transformers import Conv1D


# TODO: Organize this code into a class that fits with the rest of the syntax of the program

class SentimentalThinker(SentimentalFeatures):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embedded_sentences = np.array(self.textual_features())

    def format_data(self):
        print(self.embedded_sentences.shape)

        return self.embedded_sentences.shape

    def think(self):
        # inputs = self.format_data()

        model = keras.Sequential()
        model.add(Dense(units=2000, activation='tanh', input_dim=30))
        model.add(Dense(units=2000, activation='sigmoid'))  # Logits layers (sigmoid output of dense activation) (see: MAIT sentimental thinker)
        model.add(Embedding(output_dim=64))  # TODO: You were trying to get this model to be dimensional compdatible between the layers
        model.add(Conv1D(kernel_size=3, filters=64, activation='tanh'))
        model.add(GlobalMaxPooling1D())  # Max pooling layer (see: MAIT sentimental thinker)
        model.add(ConvLSTM2D(kernel_size=3, filters=64, activation='tanh'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(ConvLSTM2D(kernel_size=3, filters=64, activation='softmax'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Embedding(output_dim=64))  # Optional embedding layer
        model.add(SpectralNormalization(Dense(units=3500, activation='relu')))  # Spectral normalization (see: MAIT sentimental thinker)
        model.add(Dense(units=200, activation='softmax'))
        model.add(RandomFeatureGaussianProcess(units=2000))
        # model.add() # TODO: Add ConvTransformer (didn't have time to do it)
        model.add(Flatten())
        model.add(Dense(units=250, activation='tanh'))
        model.add(Dense(units=300, activation='sigmoid'))
        model.compile(loss='rmse', optimizer='adam')
        model.summary()

    def talk(self):
        COMPLETIONS_MODEL = "text-davinci-003"
        EMBEDDING_MODEL = "text-embedding-ada-002"


SentimentalThinker().think()
