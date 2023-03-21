import json
import math

import pandas as pd
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands
from ta.volume import MFIIndicator

from data_handling.data_handler import Data
from technical_analysis.tautils import *


# TODO: Add TTM Squeeze indicator, Pivot Point indicator,
# So far there is one trend-based indicator (the EMA), one volatility-based indicator (bollinger bands), one volume-based indicator (the MFI), and one momentum-based indicator
# (the MACD)

class TechnicalFeatures(Data):
    def __init__(self, oscillatory_window=14, volatility_window=14, moving_average_window=28, index_overbought=80, index_oversold=20):

        super().__init__()

        with open(r"C:\Users\nicco_ev5q3ww\OneDrive\Desktop\Market Analysis Tools\Globals.json") as f:
            settings = json.load(f).get('Technical Features Hyperparameters')

        self.oscillatory_window = settings.get("Oscillatory Window")
        self.volatility_window = settings.get("Volatility Window")
        self.moving_average_window = settings.get("Moving Average Window")
        self.std_dev_window = settings.get("Moving Average Window") // settings.get("Volatility Window")
        self.index_overbought = settings.get("Index Overbought")
        self.index_oversold = settings.get("Index Oversold")
        self.signal_period = math.ceil(settings.get("Oscillatory Window") / self.std_dev_window)
        self.technical_features = self.get_feature_values()

    def get_feature_values(self):
        technical_features = pd.DataFrame()
        technical_features['BB Moving Average'], technical_features['BB Upper'], technical_features['BB Lower'], technical_features['BB Upper Indicator'], technical_features[
            'BB Lower Indicator'] = self.bollinger_bands()
        technical_features['MFI'], technical_features['MFI Overbought'], technical_features['MFI Oversold'] = self.money_flow_index()
        technical_features['Above EMA'], technical_features['Below EMA'] = self.exponential_moving_average()
        technical_features['MACD'], technical_features['MACD Histogram'] = self.moving_average_convergence_divergence()
        technical_features = technical_features.interpolate()

        return technical_features

    def bollinger_bands(self, as_dataframe=False):

        bb_indicator = BollingerBands(self.price_data['Close'], window=self.volatility_window, window_dev=self.std_dev_window)

        if as_dataframe:

            raw_indicator_values = pd.DataFrame()

            raw_indicator_values['Moving Average'] = bb_indicator.bollinger_mavg()
            raw_indicator_values['Higher Band'] = bb_indicator.bollinger_hband()
            raw_indicator_values['Lower Band'] = bb_indicator.bollinger_lband()

            interpretive_indicator_values = pd.DataFrame()

            interpretive_indicator_values['Higher Band Indicator'] = bb_indicator.bollinger_hband_indicator()
            interpretive_indicator_values['Lower Band Indicator'] = bb_indicator.bollinger_lband_indicator()

            return raw_indicator_values, interpretive_indicator_values

        else:

            moving_average = bb_indicator.bollinger_mavg()
            upper_band = bb_indicator.bollinger_hband()
            lower_band = bb_indicator.bollinger_lband()

            higher_band_indicator = bb_indicator.bollinger_hband_indicator()
            lower_band_indicator = bb_indicator.bollinger_lband_indicator()

            return moving_average, upper_band, lower_band, higher_band_indicator, lower_band_indicator

    def money_flow_index(self, as_dataframe=False):

        mfi_indicator = MFIIndicator(high=self.price_data['High'], low=self.price_data['Low'], close=self.price_data['Close'], volume=self.price_data['Volume'],
                                     window=self.volatility_window)

        if as_dataframe:

            raw_indicator_values = pd.DataFrame()

            raw_indicator_values['Money Flow Index'] = mfi_indicator.money_flow_index()

            interpretive_indicator_values = pd.DataFrame()
            interpretive_indicator_values['Overbought Indicator'] = raw_indicator_values['Money Flow Index'].apply(mfi_overbought_indicator)
            interpretive_indicator_values['Oversold Indicator'] = raw_indicator_values['Money Flow Index'].apply(mfi_oversold_indicator)

            return raw_indicator_values, interpretive_indicator_values
        # TODO: Copy the lambda statements below into the above apply methods and delete the previous method
        else:

            mfi = mfi_indicator.money_flow_index()
            overbought_indicator = mfi.apply(lambda x: 1 if x > self.index_overbought else 0)
            oversold_indicator = mfi.apply(lambda x: 1 if x < self.index_oversold else 0)

            return mfi, overbought_indicator, oversold_indicator

    def exponential_moving_average(self, as_dataframe=False, no_indicators=False):

        ema_indicator = EMAIndicator(close=self.price_data['Close'], window=self.moving_average_window)

        if as_dataframe:
            raw_indicator_values = pd.DataFrame()

            raw_indicator_values['Exponential Moving Average'] = ema_indicator.ema_indicator()

            interpretive_indicator_values = pd.DataFrame()
            interpretive_indicator_values['Above EMA Indicator'] = raw_indicator_values.where(raw_indicator_values['Exponential Moving Average'] < self.price_data['Close'], 0)
            interpretive_indicator_values['Below EMA Indicator'] = raw_indicator_values.where(raw_indicator_values['Exponential Moving Average'] > self.price_data['Close'], 0)
            interpretive_indicator_values.where(interpretive_indicator_values == 0, 1, inplace=True)

            return raw_indicator_values, interpretive_indicator_values

        else:
            ema = ema_indicator.ema_indicator()
            above_ema = ema.where(ema < self.price_data['Close'].values, 0).where(ema == 0, 1)
            below_ema = ema.where(ema > self.price_data['Close'].values, 0).where(ema == 0, 1)

            if no_indicators:
                return ema

            else:
                return above_ema, below_ema

    def moving_average_convergence_divergence(self, as_dataframe=False):

        macd_indicator = MACD(close=self.price_data['Close'], window_slow=self.moving_average_window, window_fast=self.volatility_window, window_sign=self.signal_period)

        if as_dataframe:
            raw_indicator_values = pd.DataFrame()

            raw_indicator_values['Moving Average Convergence-Divergence'] = macd_indicator.macd()

            interpretive_indicator_values = pd.DataFrame()
            interpretive_indicator_values['MACD Histogram'] = macd_indicator.macd_diff()

            return raw_indicator_values, interpretive_indicator_values

        else:
            macd = macd_indicator.macd()
            histogram = macd_indicator.macd_diff()

            return macd, histogram
