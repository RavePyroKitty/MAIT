import json
import math

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import layers
from keras.layers import LSTM
from keras_multi_head import MultiHeadAttention

from technical_analysis.technical_features import TechnicalFeatures
# import data_handling.data_handler
from utils import data_denormalize

# TODO: Add model here that trains and exports itself similar to the method used in Tailor
# TODO: Write in the Ornstein-Uhlenbeck equations here with their respective parameters
# TODO: Include Langevin analysis for the random process within the movements of the stock prices
# See the wikipedia page: https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process with the section on financial mathematics

with open('Globals.json') as global_variables:
    variables = json.load(global_variables)


def build_random_walk_model():
    model = keras.Sequential()
    model.add(layers.Bidirectional(LSTM(units=250, input_shape=(20, 1), activation='relu', return_sequences=True)))
    model.add(layers.Dropout(0.2))
    model.add(MultiHeadAttention(LSTM(units=250, activation='tanh', return_sequences=True), num_heads=8))
    model.add(layers.Dropout(0.2))
    model.add(layers.Bidirectional(LSTM(units=350, activation='relu', return_sequences=True)))
    model.add(layers.Dropout(0.2))
    model.add(MultiHeadAttention(LSTM(units=350, activation='relu'), num_heads=8))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(units=100, activation='tanh'))
    model.add(layers.Dense(units=1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()

    return model


def train_random_walk_model(data, scaler, epochs=25, train_percentage=0.7, timesteps=20):
    data = data.fillna(value=0)
    data = data.interpolate()

    print('training data:', data.head(), data.shape)

    num_samples = data.shape[0]
    channels = data.shape[1]

    normalized_training_data = data.to_numpy()

    #   close = np.reshape(data.to_numpy(), newshape=(-1, 1))
    #   data_scaler = MinMaxScaler(copy=False)
    #   normalized_training_data = data_scaler.fit_transform(X=close)
    #    print('Training data normalized:', normalized_training_data)
    X = []
    y = []

    for i in range(num_samples - timesteps - 1):
        X.append(normalized_training_data[i:i + timesteps])
        y.append(normalized_training_data[i + timesteps + 1][0])

    X = np.array(X)
    print('X shape:', X.shape)
    X = np.reshape(X, (X.shape[0], 1, timesteps, channels))
    print('Shape of X data:', X.shape)
    y = np.array(y)
    y = np.reshape(y, (y.shape[0], 1, 1))
    print('Shape of y data:', y.shape)

    X_train, X_test = X[:math.floor(len(X) * train_percentage)], X[math.floor(len(X) * train_percentage):]
    y_train, y_test = y[:math.floor(len(y) * train_percentage)], X[math.floor(len(y) * train_percentage):]
 
    model = keras.Sequential()
    model.add(layers.Bidirectional(
        layers.ConvLSTM1D(filters=64, kernel_size=3, activation='tanh', recurrent_dropout=0.2,
                          data_format='channels_last', return_sequences=True),
        input_shape=X_train.shape[1:]))
    # model.add(layers.Dropout(0.2))
    # model.add(layers.ActivityRegularization(0.3))
    model.add(layers.Bidirectional(
        layers.ConvLSTM1D(filters=128, kernel_size=3, activation='tanh', recurrent_dropout=0.2,
                          data_format='channels_last', return_sequences=True)))
    # model.add(layers.Dropout(0.2))
    # model.add(layers.ActivityRegularization(0.3))
    model.add(layers.ConvLSTM1D(filters=64, kernel_size=3, activation='tanh', recurrent_dropout=0.2,
                                data_format='channels_last'))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=1000, activation='tanh'))
    model.add(layers.Dense(units=500, activation='softmax'))
    #  model.add(layers.Dropout(0.3))
    model.add(layers.Dense(units=100, activation='relu'))
    #  model.add(layers.Dropout(0.3))
    model.add(layers.Dense(units=32, activation='relu'))
    model.add(layers.Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=timesteps, shuffle=False)

    model.save('close_price_forecaster.h5')

    y_pred = model.predict(X_test)
    print('Prediction:', y_pred.shape)
    y_pred = y_pred.flatten()
    y = y.flatten()

    y_pred = data_denormalize(y_pred, scaler.get('Close')[0], scaler.get('Close')[1])
    y = data_denormalize(y, scaler.get('Close')[0], scaler.get('Close')[1])

    plt.plot(range(0, len(y)), y, label='True', color='Green')
    plt.plot(range(len(y) - len(y_pred), len(y)), y_pred, label='Prediction', color='Red')
    plt.title(label='Prediction + Real')
    plt.legend(loc='center left', title='Legend')
    plt.show(block=False)
    plt.pause(2)
    plt.close('all')

    return model


def random_walk_prediction(data, model, scaler, num_predictions=1, timesteps=20):
    prediction = []
    data = data.to_numpy()
    num_features = data.shape[1]
    X = data[data.shape[0] - timesteps:]

    for i in (range(num_predictions)):
        X = np.reshape(X, (1, 1, timesteps, num_features))
        y_pred = model.predict_on_batch(X)

        X = X.flatten()

        data = np.append(data, y_pred)

        prediction.append(y_pred)

    prediction = data_denormalize(prediction, scaler.get('Close')[0], scaler.get('Close')[1])
    print('Prediction denormalized:', prediction)

    return prediction


data_handler = TechnicalFeatures()
data, scaler = data_handler.get_pricing_data(normalize=True)
max_samples = data.shape[0]
technical_features = data_handler.get_feature_values()
technical_features = technical_features[:max_samples]
print('Shape of features:', technical_features.shape, 'Sample:', technical_features[:25])
data = pd.concat([data, technical_features])
data = data[:max_samples]
print('Len of data:', data.shape)

print(data.shape)

# model, scaler = train_random_walk_model(data=data['Close'])
model = train_random_walk_model(data, scaler)
prediction = random_walk_prediction(data=data, model=model, scaler=scaler)
