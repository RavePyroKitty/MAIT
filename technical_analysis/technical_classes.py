import datetime
import math

import numpy as np
import pandas as pd

from technical_analysis.technical_features import TechnicalFeatures


# TODO: Add 'as_dataframe' option to functions that return class data.

class TechnicalClasses(TechnicalFeatures):
    def __init__(self, oscillatory_window=14, volatility_window=14, moving_average_window=28, index_overbought=80, index_oversold=20, ticker='SPY', start_date=None,
                 end_date=datetime.datetime.now(), interval='15m', period='5d'):
        super().__init__(oscillatory_window=oscillatory_window, volatility_window=volatility_window, moving_average_window=moving_average_window, index_overbought=index_overbought,
                         index_oversold=index_oversold, ticker=ticker, start_date=start_date, end_date=end_date, interval=interval, period=period)
        self.difference_window = math.ceil(self.std_dev_window) if math.ceil(self.std_dev_window) <= 1 else math.ceil(self.std_dev_window)
        self.technical_classes = self.get_class_values()

    def get_class_values(self):
        technical_classes = pd.DataFrame()
        technical_classes['PPC'] = self.percent_price_change()
        technical_classes['STD DEV'] = self.standard_deviation()
        technical_classes['MEDIAN'], technical_classes['UPPER MEDIAN DEVIANCE'], technical_classes['LOWER MEDIAN DEVIANCE'] = self.intraday_median_deviation()
        technical_classes['EMA PD'] = self.moving_average_deviation()
        technical_classes['PC OF PPC'] = self.change_in_percent_price_change()
        technical_classes['PC STD DEV'], technical_classes['PC EMA PD'] = self.percent_change_in_volatility()

        return technical_classes

    def percent_price_change(self):
        return (self.price_data['Close'] / self.price_data['Open']).apply(lambda x: x / 100)

    def standard_deviation(self):
        return self.price_data['Close'].rolling(window=self.std_dev_window).std()

    def intraday_median_deviation(self):
        median = self.price_data['Open'] + np.abs(self.price_data['Close'] - self.price_data['Open'])
        day_high_distance = (self.price_data['High'] / (median + self.price_data['Open'])).apply(lambda x: x / 100)
        day_low_distance = (self.price_data['High'] / (median + self.price_data['Open'])).apply(lambda x: x / 100)

        return median, day_high_distance, day_low_distance

    def moving_average_deviation(self):
        ema = self.exponential_moving_average(no_indicators=True)
        distance_from_ema = (self.price_data['Close'] / ema).apply(lambda x: x / 100)

        return distance_from_ema

    def change_in_percent_price_change(self):
        percent_changes = self.percent_price_change().rolling(window=self.difference_window).apply(lambda x: x.iloc[0] - x.iloc[self.difference_window - 1])

        return percent_changes

    def percent_change_in_volatility(self):
        standard_dev = self.standard_deviation().rolling(window=self.difference_window).apply(lambda x: (x.iloc[0] - x.iloc[self.difference_window - 1]) / 100)
        ema_deviance = self.moving_average_deviation().rolling(window=self.difference_window).apply(lambda x: (x.iloc[0] - x.iloc[1]) / 100)

        return standard_dev, ema_deviance
