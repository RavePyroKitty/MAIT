import numpy as np

from data_handler import data_preprocess

# TODO: Have the program self update the batch variable (and others if needed) based on the Hurst Exponent
#  (batch should equal timesteps for hurst = 0.5 if model is RW)


def get_hurst_exponent(data, lag=20):  # Calculate Hurst Exponent

    lags = range(2, lag)
    tau = [np.std(np.subtract(data[lag:], data[:-lag])) for lag in lags]
    reg = np.polyfit(np.log(lags), np.log(tau), 1)

    return reg[0]


def calculate_volatility(data, period=30):
    data = data['Close']
    data = data_preprocess(data=data, batch_size=period, normalize=False)
    mean = np.array(np.nan)
    volatility = np.array(np.nan)

    for i in range(len(data)):
        if i * period == 0:
            pass

        mean = np.append(np.sum(data[(i * period) - period:(i * period)]))
        volatility = np.append(np.log(data[(i * period)]).std())
        volatility = np.log(data).std() * np.sqrt(period)

    return volatility


def ornstein_uhlenbeck(data):
    data
    volatility = 0  # Define this
    mean = 0  # Define this
    theta = 0  # Define this

    return None


# Calculate exponent using lags
"""for lag in [20, 100, 300, 500, 1000]:
    hurst_exp = get_hurst_exponent(data["Close"].values, lag)
    print(f"Hurst exponent with {lag} lags: {hurst_exp:.4f}")"""

# Simple RW algorithm for when Hurts = 0.5:
# Timesteps should be the same as the lag of the hurst that generated 0.5

# print('Future prediction:')
# print(RW(data))
