import alpaca_trade_api as alpaca
import requests


def alpaca_authorization(live=False):

    if live:
        secret_key = "IBAAAUbhmCvqUeZ5vSxDouyFcDC5bCJZeFqmDTcX"
        api_key = "AKO8114OKV3WOL1MFYUP"
        endpoint = "https://api.alpaca.markets"

    else:
        secret_key = "PKPS6TXSJTYANGU432LG"
        api_key = "Dw4JzCGDQ604GJDig7Ev1Cp49mlVrCdc4cvfGE9x"
        endpoint = "https://paper-api.alpaca.market"

    rest = alpaca.REST(key_id=api_key, secret_key=secret_key, base_url=endpoint)

    return rest


def td_ameritrade_authorization():
    authorization_code = "xbAK9ONFI2uifXjSllPAD5Kf8i9OMhT2KpHCG28hh4VB1NPoj1L++wVo+GaM1fuPFoJCla1iDHD9x5mRQQXpiBs05xNZ8NAINEX+JTl6crLGuRwG6q2vnNPiGuYnr0ihGHvti8Vz7L2nhCEI4ylc2HGZPa0Ja6oa6wVEcbCM45XP+s8vu0ylOxo4Q9IpSQY+zS4eTVfR5MYX9f3f3Yf6LevY/7fTq3f/Ga8qfCsypj2JmV57VJtTvWVQC6rYQqR1vrkpgUg+psvV+aLR2sc5EYzDvzmELdZwtF3eQI4HAC9w0sZOTxfPSDEFBeRSe0XAiVYdNN9hfSJZCdzFFn2SCmHlsOOJq7asiE/cl/u2gRJefe8GEwr0ND3NK8BTCrY+kvysTg3JI62Qhm/GjGM2dEqYOZE/wjm9NnbP7cwoStJdDjrzvdDHiiqgLrQ100MQuG4LYrgoVi/JHHvlcnvh+6tVhUuNCrkb2ACU3MAofy7ev4v9E9exiCa6LWuioZJdLrrEijRjeaLpcRVT4OGQOgsrfoGxCHiaRSGtocLvTNhCe//YevJZH1Xxhs/uVjLaeXf795U39fW0bskgjFn99HP16TNMI1FXGR1H6stsXuv/jXRHQEGH1twCGY6gEg7uDJpR58fThf3lNSpEAelwF8au9tMpQb0Wr/8wC3iFftiGsX5j8WJLpQw47ldghK2KqissCZ1UWTBXwOtRqdIdLc58rs3TqbExlOSONp7HUeFVBCWwCKJIgkfnmrEKYjn16c5IYgEXAVXHA0FMZXw71Hb8lIxEWZDo7Mgt/R6VDwsvjLXHINnbsQniln170iSAEfiYyLwaLimGLE62jqkS9QO1grrfx6OU5gUs6mDwP+Vqwu/+MhjYMqjW7x4YTI7KBFGi/B7PWW4=212FD3x19z9sWBHDJACbC00B75E"
    grant_type = 'authorization_code'
    consumer_key = 'CQDBO4LWTWC0ZJJOMLNHEV9UEQMVCFFJ'

    r = requests.post(f"https://api.tdameritrade.com/v1/oauth2/token?authorization_code={authorization_code}&?"
                      f"grant_type={grant_type}&?client_id={consumer_key}")

    r = r.json()
    print(r)

    return None
