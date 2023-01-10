import json
import math

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

import data_handling.data_handler
from data_handling.data_handler import data_denormalize, data_preprocess

# TODO: Add model here that trains and exports itself similar to the method used in Tailor
# TODO: Write in the Ornstein-Uhlenbeck equations here with their respective parameters
# TODO: Include Langevin analysis for the random process within the movements of the stock prices
# See the wikipedia page: https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process with the section on financial mathematics

with open('Globals.json') as global_variables:
    variables = json.load(global_variables)


def random_walk_grid_search(model=None, X=None, y=None, parameters=variables["Hyperparameter Grid"]):
    model = KerasClassifier(build_fn=build_random_walk_model)

    grid_search = GridSearchCV(estimator=model, param_grid=parameters, cv=5)

    grid_search = grid_search.fit(X=X, y=y)

    best_params = grid_search.best_params_
    accuracy = grid_search.best_score_

    return best_params, accuracy


def build_random_walk_model(units=variables["Model Variables"]["Units"],
                            activation=variables["Model Variables"]["Activation"],
                            loss=variables["Model Variables"]["Loss"],
                            optimizer=variables["Model Variables"]["Optimizer"],
                            timesteps=20, features=1):
    model = keras.Sequential()
    model.add(layers.LSTM(units=units, input_shape=(timesteps, features), activation='relu', return_sequences=True))
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(units=250, activation='tanh', return_sequences=True))
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(units=350, activation='relu', return_sequences=True))
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(units=350, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(units=100, activation='tanh'))
    model.add(layers.Dense(units=1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    model.summary()

    return model


def train_random_walk_model(data, epochs=variables["Model Variables"]["Epochs"],
                            train_percentage=variables["Model Variables"]["Training Percentage"],
                            timesteps=20, gs_params=variables["Hyperparameter Grid"], grid_search=False):
    data = data_preprocess(data=data, batch_size=timesteps)
    data_max = data['Close'].max()
    data_min = data['Close'].min()
    data = data_preprocess(data=data, normalize=True, batch_size=timesteps)
    close = data['Close']

    X = []
    y = []

    for i in range(len(data) - timesteps):
        X.append(close[i:i + timesteps])
        y.append(close[i + timesteps])

    X = np.array(X)
    X = np.reshape(X, (X.shape[0], timesteps, 1))
    print('Shape of X data:')
    print(X.shape)
    print('X data:', X)
    y = np.array(y)
    y = np.reshape(y, (y.shape[0], 1))
    print('Shape of y data:')
    print(y.shape)

    # Split into training and testing sets for both the X and y datasets
    X_train, X_test = X[:math.floor(len(X) * train_percentage)], X[math.floor(len(X) * train_percentage):]
    y_train, y_test = y[:math.floor(len(y) * train_percentage)], X[math.floor(len(y) * train_percentage):]

    # If this is a new classification, the following code is for the hyperparamter grid search
    if grid_search == True:
        print('gs_shape:')
        print((X_train.shape[1], X_train.shape[2]))

        # Run a grid search using sklearn and return the paramters and accuracy
        best_params, accuracy = random_walk_grid_search(X=X_train, y=y_train, parameters=gs_params)
        print(accuracy)
        print(best_params)

        # Build a model using the previously found parameters
        model = build_random_walk_model(units=best_params['units'], epochs=best_params['epochs'],
                                        activation=best_params['activation'], loss=best_params['loss'],
                                        optimizer=best_params['optimzer'],
                                        input_shape=(X_train.shape[1], X_train.shape[2]))

    else:
        # Build a model using default parameters (see the corresping function definition for these)
        model = build_random_walk_model()

    # Fit model and make predictions
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=timesteps, shuffle=False)
    y_pred = model.predict(X_test)
    y_pred = y_pred.flatten()

    # Visualize training
    plt.plot(history.history['loss'], label='Loss')
    plt.title(label='Loss')
    plt.legend(loc='center right', title='Legend')
    plt.show(block=False)
    plt.pause(2)
    plt.close('all')

    # Postprocess data
    y_pred = data_denormalize(data=y_pred, min=data_min, max=data_max)
    y = y.flatten()
    y = data_denormalize(data=y, min=data_min, max=data_max)

    # Visualize prediction
    plt.plot(range(0, len(y)), y, label='True', color='Green')
    plt.plot(range(len(y) - len(y_pred), len(y)), y_pred, label='Prediction', color='Red')
    plt.title(label='Prediction + Real')
    plt.legend(loc='center left', title='Legend')
    plt.show(block=False)
    plt.pause(2)
    plt.close('all')

    # Write a function that makes plotting timeseries data super easy (i.e. handles dates/times on the X axis)
    y_test = y_test.flatten()

    return model, X, data_min, data_max, y_test


def random_walk_prediction(data=None, model=None, num_predictions=variables["Model Variables"]["Number of " \
                                                                                               "Predictions"],
                           batch_size=variables["Model Variables"]["Batch Size"], data_min=None, data_max=None):
    prediction = []
    # prediction = np.array(prediction)

    data = np.array(data)
    data = data.flatten()
    X = data[len(data) - batch_size:]

    for i in (range(num_predictions)):
        X = np.reshape(X, (1, batch_size, 1))

        print('Shape of input data:')
        print(X.shape)

        # y_pred = model.predict(X, batch_size=batch_size)
        y_pred = model.predict_on_batch(X)

        print('Prediction:', y_pred)

        X = X.flatten()
        X = np.delete(X, 0)
        X = np.append(X, y_pred.flatten())

        data = np.append(data, y_pred)

        prediction.append(y_pred)

    # prediction = model.predict(X)
    # prediction = prediction.flatten()
    # prediction = data_denormalize(prediction, min=data_min, max=data_max)

    prediction = data_denormalize(prediction, min=data_min, max=data_max)
    print('Prediction denormalized:', prediction)

    return prediction


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


data = data_handling.data_handler.Data().get_pricing_data()

RW(data=data)
