import datetime

import yfinance as yf
from newsapi import NewsApiClient
from newspaper import Article


class Data:
    def __init__(self, ticker='SPY', start_date=None, end_date=datetime.datetime.now(), interval='15m', period='5d', news_outlets=None):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.period = period
        self.price_data = self.get_pricing_data()
        self.newsapi = NewsApiClient(api_key='5376f110fee146ba838c8e92e115fdba')

    def get_pricing_data(self):  # Date input format = "year-month-day"

        start = self.start_date
        end = self.end_date

        if self.start_date is None:
            start = self.end_date - datetime.timedelta(days=5)
            start = str(start.strftime("%Y-%m-%d"))
            end = str(self.end_date.strftime("%Y-%m-%d"))

        ticker = yf.Ticker(ticker=self.ticker)
        price_data = ticker.history(start=start, end=end, interval=self.interval, period=self.period)

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


print(Data().get_news_articles())
