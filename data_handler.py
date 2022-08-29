import json

import matplotlib.pyplot as plt
import yfinance as yf

with open("Globals.json") as json_data_file:
    variables = json.load(json_data_file)


class DataHandler:
    def __init__(self):
        self.ticker = variables['Data']['Ticker']
        self.start_date = variables['Data']['Start Date']
        self.end_date = variables['Data']['End Date']
        self.batch_size = variables['Model Variables']['Batch Size']

    def get_yf_data(self, verbose=False):
        data = yf.download(tickers=self.ticker, start=self.start_date, end=self.end_date)

        if verbose:
            plt.plot(data['Close'], color='blue', label='Closing Prices')
            plt.title(label='Raw data')
            plt.legend(loc='center left', title='Legend')
            plt.show(block=False)
            plt.pause(2)
            plt.close('all')

        return data


# OPERATIONS:

##################################################


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
