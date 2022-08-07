from cgi import print_environ_usage
from distutils.command.build import build
from pyexpat import features
from time import time   
from turtle import color
from matplotlib import units
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
import pandas as pd
import keras
from keras import layers
import numpy as np
import math
from sklearn import datasets
from torch import batch_norm_stats, lstm
import matplotlib.pyplot as plt
import sklearn
from keras.wrappers.scikit_learn import KerasClassifier
import yfinance as yf
import json

#TODO:
#Have the script self update the batch variable (and others if needed) based on the Hurst Exponent (batch should equal timesteps for hurst = 0.5 if 
# model is RW)

#Loading global variables
with open("Globals.json") as json_data_file:
    variables = json.load(json_data_file)
print(variables)

print('Model Variables:')
print(variables['Model Variables'])

print('Data Variables:')
print(variables['Data'])

print('Hyperparameter Grid Space:')
print(variables['Hyperparameter Grid'])

#Load dataset
data = yf.download(variables['Data']['Ticker'], start=variables['Data']['Start Date'], end=variables['Data']['End Date'])
print('Data head:')
print(data.head())
print(data)

#Plot Data
plt.plot(data['Close'], color='blue', label='Closing Prices')
plt.title(label='Raw data')
plt.legend(loc='center left', title='Legend')
plt.show(block=False)
plt.pause(2)
plt.close('all')

#Clean data a lil
print('Data Cleaned:')
print(data.head())

#Data Trim
def data_preprocess(data, batch):    
    remainder = int(len(data))%batch
    data = data[:-remainder]
    
    return data

#Data Normalize (min-max)
def data_normalize(data):    
    normalized = (data - data.min())/(data.max() - data.min())
    
    return normalized

#Data Denormalize (min-max)
def data_denormalize(data, min, max):
    denormalized = []
    
    for i in(range(len(data))):
        denormalized.append((data[i]*(max - min)) + min)
    
    return denormalized

#Calculate Hurst Exponent
def get_hurst_exponent(data, lag=20):
    lags = range(2, lag)
    tau = [np.std(np.subtract(data[lag:], data[:-lag])) for lag in lags]    
    reg = np.polyfit(np.log(lags), np.log(tau), 1)

    return reg[0]

#Calculate exponenet using lags 
"""for lag in [20, 100, 300, 500, 1000]:
    hurst_exp = get_hurst_exponent(data["Close"].values, lag)
    print(f"Hurst exponent with {lag} lags: {hurst_exp:.4f}")"""
        
#Simple RW algorithm for when Hurts = 0.5:
#Timesteps should be the same as the lag of the hurst that generated 0.5

def random_walk_grid_search(model=None, X=None, y=None, parameters=variables["Hyperparameter Grid"]):
    model = KerasClassifier(build_fn=build_random_walk_model)
    
    grid_search = GridSearchCV(estimator=model, param_grid=parameters, cv=5)
    
    grid_search = grid_search.fit(X=X, y=y)
    
    best_params = grid_search.best_params_
    accuracy = grid_search.best_score_
    
    return best_params, accuracy

def build_random_walk_model(units=variables["Model Variables"]["Units"], activation=variables["Model Variables"]["Activation"],
                            loss=variables["Model Variables"]["Loss"], optimizer=variables["Model Variables"]["Optimizer"],
                            timesteps=20, features=1):
    model = keras.Sequential()
    model.add(layers.LSTM(units=units, input_shape=(timesteps, features), activation=activation))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(units=1))
    model.compile(loss=loss, optimizer=optimizer)
    model.summary()
    
    return model
                
def train_random_walk_model(data, epochs=variables["Model Variables"]["Epochs"],
                            train_percentage=variables["Model Variables"]["Training Percentage"],
                            timesteps=20, gs_params=variables["Hyperparameter Grid"], grid_search=False):
    
    #Preprocess data
    data = data_preprocess(data=data, batch=timesteps)
    data_max = data['Close'].max()
    data_min = data['Close'].min()
    data = data_normalize(data=data)
    close = data['Close']
    
    X = []
    y = []
    
    #Honestly don't think this is necessary, but why not. We're using the past group of n timesteps to as the single feature to predict the next datapoint
    for i in range(len(data) - timesteps):
        X.append(close[i:i+timesteps])
        y.append(close[i+timesteps])
    
    #Shape input data for model training [Number of samples (length of dataset),  batch size (timesteps), number of features (1)]
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], timesteps, 1))
    print('Shape of X data:')
    print(X.shape)
    y = np.array(y)
    y = np.reshape(y, (y.shape[0], 1))
    print('Shape of y data:')
    print(y.shape)
    
    #Split into training and testing sets for both the X and y datasets
    X_train, X_test = X[:math.floor(len(X)*train_percentage)], X[math.floor(len(X)*train_percentage):]
    y_train, y_test = y[:math.floor(len(y)*train_percentage)], X[math.floor(len(y)*train_percentage):]
   
    #If this is a new classification, the following code is for the hyperparamter grid search
    if grid_search == True:
        print('gs_shape:')
        print((X_train.shape[1], X_train.shape[2]))
        
        #Run a grid search using sklearn and return the paramters and accuracy
        best_params, accuracy = random_walk_grid_search(X=X_train, y=y_train, parameters=gs_params)
        print(accuracy)
        print(best_params)
        
        #Build a model using the previously found parameters
        model = build_random_walk_model(units=best_params['units'], epochs=best_params['epochs'], 
                                activation=best_params['activation'],  loss=best_params['loss'], optimizer=best_params['optimzer'],
                                input_shape=(X_train.shape[1], X_train.shape[2]))
    
    else:
        #Build a model using default parameters (see the corresping function definition for these)
        model = build_random_walk_model()
    
    #Fit model and make predictions
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=timesteps, shuffle=False)
    y_pred = model.predict(X_test)
    y_pred = y_pred.flatten()
    
    #Visualize training
    plt.plot(history.history['loss'], label='Loss')
    plt.title(label='Loss')
    plt.legend(loc='center right', title='Legend')
    plt.show(block=False)
    plt.pause(2)
    plt.close('all')
    
    #Postprocess data
    y_pred = data_denormalize(data=y_pred, min=data_min, max=data_max)
    y = y.flatten()
    y = data_denormalize(data=y, min=data_min, max=data_max)
    
    #Visualize prediction
    plt.plot(range(0, len(y)), y, label='True', color='Green')
    plt.plot(range(len(y) - len(y_pred), len(y)), y_pred, label='Prediction', color='Red')
    plt.title(label='Prediction + Real')
    plt.legend(loc='center left', title='Legend')
    plt.show(block=False)
    plt.pause(2)
    plt.close('all')
    
    #Write a function that makes plotting timeseries data super easy (i.e. handles dates/times on the X axis)
    y_test = y_test.flatten()
    
    return model, X, data_min, data_max, y_test

def random_walk_prediction(data=None, model=None, num_predictions=variables["Model Variables"]["Number of Predictions"],
                           batch_size=variables["Model Variables"]["Batch Size"], data_min=None, data_max=None):
    prediction = []
    prediction = np.array(prediction)
    
    data = np.array(data)
    data = data.flatten()
    X = data[(len(data) - batch_size):]
    X = np.reshape(X, (1, batch_size, 1))
    
    for i in(range(num_predictions)):
        
        print('Shape of input data:')
        print(X.shape)
        
        y_pred = model.predict(X, batch_size=batch_size)
        
        X = X.flatten()
        X = np.append(X, (data[i:i+batch_size]))
        X = np.reshape(X, (i+2, batch_size, 1))
        
        
        data = np.append(data, y_pred)
    
        y_pred = data_denormalize(y_pred, min=data_min, max=data_max)
        prediction = np.append(prediction, y_pred)
    
    prediction = model.predict(X)
    prediction = prediction.flatten()
    prediction = data_denormalize(prediction, min=data_min, max=data_max)
    
    
    return prediction

def RW(data=None, epochs=variables["Model Variables"]["Epochs"], train_percentage=variables["Model Variables"]["Training Percentage"],
       gs_params=variables["Hyperparameter Grid"], timesteps=20):

    model, X, data_min, data_max, y_test = train_random_walk_model(data=data, timesteps=timesteps, epochs=epochs, 
                                                                  train_percentage=train_percentage, gs_params=gs_params)
    prediction = random_walk_prediction(data=X, model=model, batch_size=timesteps, data_min=data_min, data_max=data_max)
    
    #Plot the previous batch in one color and the prediction in another for better visualization

    #plt.plot(previous.shape[0], previous, label='True Test Data', color='red')
    #plt.plot((previous.shape[0] + prediction.shape[0]), prediction, label='Prediction', color='blue')
    plt.plot(prediction, color='red', label='prediction')
    plt.title(label='Future prediction')
    plt.legend(loc='center left', title='Legend')
    plt.show()
    
    return prediction

#print('Future prediction:')
#print(RW(data))


def calculate_volatility(data, period=30):
    
    data = data['Close']
    data = data_preprocess(data=data, batch=period)
    mean = np.array(np.nan)
    volatility = np.array(np.nan)
    
    for i in range(len(data)):
        if i*period == 0:
            pass
        
        mean = np.append(np.sum(data[(i*period)-period:(i*period)]))
        volatility = np.append(np.log(data[(i*period)]).std())
        volatility = np.log(data).std()*np.sqrt(period)
        
    return volatility

print(calculate_volatility(data))


def ornstein_uhlenbeck(data):
    data
    volatility = 0 #Define this
    mean = 0 #Define this
    theta = 0 #Define this

#TODO

#Write in the Ornstein-Uhlenbeck equations here with their respective parameters. Include Langevin alaysis for the random
#process within the movements of the stock prices
#See the wikipedia page: https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process with the section on financial mathematics

    return None
