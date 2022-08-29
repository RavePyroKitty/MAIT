from tda import auth, client

# TODO: Figure out trading interface for both crypto and stocks
# TODO: Set up TD Ameritrade API

code = xbAK9ONFI2uifXjSllPAD5Kf8i9OMhT2KpHCG28hh4VB1NPoj1L + +wVo + GaM1fuPFoJCla1iDHD9x5mRQQXpiBs05xNZ8NAINEX + JTl6crLGuRwG6q2vnNPiGuYnr0ihGHvti8Vz7L2nhCEI4ylc2HGZPa0Ja6oa6wVEcbCM45XP + s8vu0ylOxo4Q9IpSQY + zS4eTVfR5MYX9f3f3Yf6LevY / 7
fTq3f / Ga8qfCsypj2JmV57VJtTvWVQC6rYQqR1vrkpgUg + psvV + aLR2sc5EYzDvzmELdZwtF3eQI4HAC9w0sZOTxfPSDEFBeRSe0XAiVYdNN9hfSJZCdzFFn2SCmHlsOOJq7asiE / cl / u2gRJefe8GEwr0ND3NK8BTCrY + kvysTg3JI62Qhm / GjGM2dEqYOZE / wjm9NnbP7cwoStJdDjrzvdDHiiqgLrQ100MQuG4LYrgoVi / JHHvlcnvh + 6
tVhUuNCrkb2ACU3MAofy7ev4v9E9exiCa6LWuioZJdLrrEijRjeaLpcRVT4OGQOgsrfoGxCHiaRSGtocLvTNhCe // YevJZH1Xxhs / uVjLaeXf795U39fW0bskgjFn99HP16TNMI1FXGR1H6stsXuv / jXRHQEGH1twCGY6gEg7uDJpR58fThf3lNSpEAelwF8au9tMpQb0Wr / 8
wC3iFftiGsX5j8WJLpQw47ldghK2KqissCZ1UWTBXwOtRqdIdLc58rs3TqbExlOSONp7HUeFVBCWwCKJIgkfnmrEKYjn16c5IYgEXAVXHA0FMZXw71Hb8lIxEWZDo7Mgt / R6VDwsvjLXHINnbsQniln170iSAEfiYyLwaLimGLE62jqkS9QO1grrfx6OU5gUs6mDwP + Vqwu / +MhjYMqjW7x4YTI7KBFGi / B7PWW4 = 212
FD3x19z9sWBHDJACbC00B75E


# probably refresh token

class Interface:
    def __init__(self):
        self.interface = 'TD Ameritrade'

    def get_stock_data(self):
        return None


token_path = '/path/to/token.json'
api_key = 'YOUR_API_KEY@AMER.OAUTHAP'
redirect_uri = 'https://your.redirecturi.com'
try:
    c = auth.client_from_token_file(token_path, api_key)
except FileNotFoundError:
    from selenium import webdriver

    with webdriver.Chrome() as driver:
        c = auth.client_from_login_flow(
            driver, api_key, redirect_uri, token_path)

r = c.get_price_history('AAPL',
                        period_type=client.Client.PriceHistory.PeriodType.YEAR,
                        period=client.Client.PriceHistory.Period.TWENTY_YEARS,
                        frequency_type=client.Client.PriceHistory.FrequencyType.DAILY,
                        frequency=client.Client.PriceHistory.Frequency.DAILY)
assert r.status_code == 200, r.raise_for_status()
print(json.dumps(r.json(), indent=4))
