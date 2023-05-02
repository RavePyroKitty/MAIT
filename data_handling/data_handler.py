import datetime
import json

import yfinance as yf
from newsapi import NewsApiClient
from newspaper import Article
from sklearn.preprocessing import StandardScaler

# with open("Globals.json") as json_data_file:
#    variables = json.load(json_data_file)

"""
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
        stock_data = alpaca_rest.get_bars(symbol=ticker, timeframe='day', start=start_date, end=end_date)
        stock_data = pd.DataFrame(data=stock_data)
        # TODO: Make the timeframe argument work so that the REST can get the stock bars
        if verbose:
            plt.plot(stock_data['c'], color='blue', label='Closing Prices')
            plt.title(label='Raw data')
            plt.legend(loc='center left', title='Legend')
            plt.show(block=False)
            plt.pause(2)
            plt.close('all')

        return stock_data
"""


# OPERATIONS:

##################################################

def format(data):
    # Format the incoming data from multiple sources into the same format of data for model training

    return None


# Data Denormalize (min-max)

class Data:
    def __init__(self, ticker='SPY', start_date=None, end_date=datetime.datetime.now(), interval='1d', period='1mo',
                 news_outlets=None):
        with open(r"C:\Users\nicco_ev5q3ww\OneDrive\Desktop\Market Analysis Tools\Globals.json") as f:
            settings = json.load(f).get('Pricing Metadata')
        with open(r"C:\Users\nicco_ev5q3ww\OneDrive\Desktop\Market Analysis Tools\Globals.json") as f:
            keys = json.load(f).get('Keys')

        self.ticker = settings.get('Ticker')
        self.start_date = settings.get('Start Date')
        self.end_date = settings.get('End Date')
        self.interval = settings.get('Interval')
        self.period = settings.get('Period')
        self.price_data = self.get_pricing_data()
        self.newsapi = NewsApiClient(api_key=keys.get("NewsAPI"))

    def get_pricing_data(self, normalize=False):  # Date input format = "year-month-day" # TODO: Add 'normalized' feature to it to
        # return normalized data

        start = self.start_date
        end = self.end_date

        if self.start_date is None:
            start = self.end_date - datetime.timedelta(days=25)
            start = str(start.strftime("%Y-%m-%d"))
            end = str(self.end_date.strftime("%Y-%m-%d"))

        ticker = yf.Ticker(ticker=self.ticker)
        price_data = ticker.history(start=start, end=end, interval=self.interval, period=self.period)

        print('Sample pricing data:', price_data.head())
        if normalize:
            scaler = StandardScaler()
            scaler.fit(price_data)
            scaled_values = scaler.transform(price_data)
            # price_data, scaling_values = normalizer(price_data)
            print('Sample normalized pricing data:', price_data[:10])
            return scaled_values, scaler

        return price_data

    def get_news_articles(self):  # TODO: Ensure that this is using a wide variety of queries relevant to the market and is able to identify those queries by reading articles
        # from existing queries (i.e. should learn what it needs to look up the more it looks stuff up (maybe it sees a similar word often so it adds that to the query list, etc))

        # TODO: News article param is passed into this function from class definition
        news_articles_dict = self.newsapi.get_everything(q='stock market')

        article_urls = [(i['url']) for i in news_articles_dict.get('articles')]

        print('Num articles:', len(article_urls))

        article_content = []

        for url in article_urls[:3]:
            article = Article(url)
            article.download()
            article.parse()
            content = article.text
            article_content.append(content)

        print(article_content)

        return article_content
