import json
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from authorization import alpaca_authorization

with open("Globals.json") as json_data_file:
    variables = json.load(json_data_file)


def get_raw_data(ticker, start_date, end_date, source='yahoo_finance', verbose=False):

    if source == 'yahoo_finance':

        data = yf.download(tickers=ticker, start=start_date, end=end_date)

        if verbose:
            plt.plot(data['Close'], color='blue', label='Closing Prices')
            plt.title(label='Raw data')
            plt.legend(loc='center left', title='Legend')
            plt.show(block=False)
            plt.pause(2)
            plt.close('all')

        return data

    if source == 'alpaca':
        alpaca_rest = alpaca_authorization()
        stock_data = alpaca_rest.get_bars(symbol=ticker, start=start_date, end=end_date)
        stock_data = pd.DataFrame(data=stock_data)

        return stock_data


# OPERATIONS:

##################################################

def format():

    # Format the incoming data from multiple sources into the same format of data for model training

    return None

def data_preprocess(data, batch_size, normalize=True):
    remainder = int(len(data)) % batch_size
    data = data[:-remainder]

    if normalize:
        data = (data - data.min()) / (data.max() - data.min())

    data.fillna(value=0, inplace=True)

    return data


# Data Denormalize (min-max)
def data_denormalize(data, min, max):
    denormalized = []

    for i in (range(len(data))):
        denormalized.append((data[i] * (max - min)) + min)

    return denormalized
