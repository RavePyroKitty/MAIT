import requests

from data_handling.data_handler import Data


class SentimentalClasses(Data):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_quarterly_gdp(self):
        url = 'https://www.alphavantage.co/query?function=REAL_GDP&interval=annual&apikey=AFVC8059TYVTZNWS'
        r = requests.get(url)
        json_data = r.json()
        data_as_dict = json_data.get('data')

        gdp = []

        for value in data_as_dict:
            gdp.append(value.get('value'))

        return gdp

    def get_treasury_yeild(self, interval='monthly', maturity='10year'):
        url = f"https://www.alphavantage.co/query?function=TREASURY_YIELD&interval={interval}&maturity={maturity}&apikey=AFVC8059TYVTZNWS"
        r = requests.get(url)
        json_data = r.json()
        data_as_dict = json_data.get('data')

        treasury_yeild = []

        for value in data_as_dict:
            treasury_yeild.append(value.get('value'))

        print(treasury_yeild)

        return treasury_yeild

    def get_cpi(self, interval='monthly'):
        url = f"https://www.alphavantage.co/query?function=CPI&interval={interval}&apikey=AFVC8059TYVTZNWS'"
        r = requests.get(url)
        json_data = r.json()
        data_as_dict = json_data.get('data')

        cpi_data = []

        for value in data_as_dict:
            cpi_data.append(value.get('value'))

        print(cpi_data)

        return cpi_data


print(SentimentalClasses().get_cpi())
