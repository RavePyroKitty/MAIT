import json
import math

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import data_handling.data_handler
from data_handling.data_handler import data_denormalize, data_preprocess

# TODO: Add model here that trains and exports itself similar to the method used in Tailor
# TODO: Write in the Ornstein-Uhlenbeck equations here with their respective parameters
# TODO: Include Langevin analysis for the random process within the movements of the stock prices
# See the wikipedia page: https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process with the section on financial mathematics

with open('Globals.json') as global_variables:
    variables = json.load(global_variables)


def build_random_walk_model():
    model = keras.Sequential()
    model.add(layers.LSTM(units=250, input_shape=(20, 1), activation='relu', return_sequences=True))
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(units=250, activation='tanh', return_sequences=True))
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(units=350, activation='relu', return_sequences=True))
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(units=350, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(units=100, activation='tanh'))
    model.add(layers.Dense(units=1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()

    return model


def train_random_walk_model(data, epochs=50, train_percentage=0.7, timesteps=20):

    print('training data:', data.head())

    close = np.reshape(data.values, newshape=(-1, 1))
    data_scaler = MinMaxScaler(copy=False)
    normalize_training_data = data_scaler.fit_transform(X=close)

    X = []
    y = []

    for i in range(len(normalize_training_data) - timesteps):
        X.append(close[i:i + timesteps])
        y.append(close[i + timesteps])

    X = np.array(X)
    X = np.reshape(X, (X.shape[0], timesteps, 1))
    print('Shape of X data:', X.shape)
    y = np.array(y)
    y = np.reshape(y, (y.shape[0], 1, 1))
    print('Shape of y data:', y.shape)

    X_train, X_test = X[:math.floor(len(X) * train_percentage)], X[math.floor(len(X) * train_percentage):]
    y_train, y_test = y[:math.floor(len(y) * train_percentage)], X[math.floor(len(y) * train_percentage):]

    model = build_random_walk_model()

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=timesteps, shuffle=False)
    y_pred = model.predict(X_test)
    y_pred = y_pred.flatten()

    y_pred = np.reshape(y_pred, newshape=(-1, 1))
    y = np.reshape(y, newshape=(-1, 1))

    y_pred = data_scaler.inverse_transform(y_pred)
    y = data_scaler.inverse_transform(y)

    plt.plot(range(0, len(y)), y, label='True', color='Green')
    plt.plot(range(len(y) - len(y_pred), len(y)), y_pred, label='Prediction', color='Red')
    plt.title(label='Prediction + Real')
    plt.legend(loc='center left', title='Legend')
    plt.show(block=False)
    plt.pause(2)
    plt.close('all')

    y_test = y_test.flatten()

    return model, data_scaler


def random_walk_prediction(data, model, scaler, num_predictions=30):
    prediction = []
    data = np.array(data)
    data = data.flatten()
    X = data[len(data) - num_predictions:]

    for i in (range(num_predictions)):
        X = np.reshape(X, (1, num_predictions, 1))

        print('Shape of input data:', X.shape)

        y_pred = model.predict_on_batch(X)

        X = X.flatten()
        X = np.delete(X, 0)
        X = np.append(X, y_pred.flatten())

        data = np.append(data, y_pred)

        prediction.append(y_pred)

    prediction = scaler.inverse_transform(prediction)
    print('Prediction denormalized:', prediction)

    return prediction

data = data_handling.data_handler.Data().get_pricing_data()

model, scaler = train_random_walk_model(data=data['Close'])



def RW(data=None, epochs=variables["Model Variables"]["Epochs"],
       train_percentage=variables["Model Variables"]["Training Percentage"],
       gs_params=variables["Hyperparameter Grid"], timesteps=20):
    model, X, data_min, data_max, y_test = train_random_walk_model(data=data, timesteps=timesteps, epochs=epochs,
                                                                   train_percentage=train_percentage,
                                                                   gs_params=gs_params)
    prediction = random_walk_prediction(data=X, model=model, batch_size=timesteps, data_min=data_min, data_max=data_max)

    # Plot the previous batch in one color and the prediction in another for better visualization

    # plt.plot(previous.shape[0], previous, label='True Test Data', color='red')
    # plt.plot((previous.shape[0] + prediction.shape[0]), prediction, label='Prediction', color='blue')
    plt.plot(prediction, color='red', label='prediction')
    plt.title(label='Future prediction')
    plt.legend(loc='center left', title='Legend')
    plt.show()

    return prediction

def random_walk_grid_search(model=None, X=None, y=None, parameters=variables["Hyperparameter Grid"]):
    model = KerasClassifier(build_fn=build_random_walk_model)

    grid_search = GridSearchCV(estimator=model, param_grid=parameters, cv=5)

    grid_search = grid_search.fit(X=X, y=y)

    best_params = grid_search.best_params_
    accuracy = grid_search.best_score_

    return best_params, accuracy
