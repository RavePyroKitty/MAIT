# TODO: All inputted variables should have algorithmic determinations (stop_loss, stop_price, etc.) based on the
#  decisions to submit them (also produced in the algorithm)

class Orders:
    def __init__(self, brokerage):
        self.brokerage = brokerage

    def buy(self, ticker, amount, take_profit=None, stop_loss=None, stop_price=None):

        # TODO: "Amount" variable should handle converting inputted dollar amounts into share quantities and a limit price
        #  for input into the order JSON

        if self.brokerage == 'ALPACA':
            # TODO: This right now is a bracket order, update so that the function handles NOT submitting a bracket order if there
            #  is no take_profit/stop_loss/stop_price inputted. (Bracket order means a buy order is submitted simultaneously with a sell order at the
            #  take_profit price and a sell order to trigger when the price hits the stop loss price to sell at the limit price)
            buy = {
                "side": "buy",
                "symbol": ticker,
                "type": "market",
                "qty": amount,
                "time_in_force": "gtc",
                "order_class": "bracket",
                "take_profit": {
                    "limit_price": take_profit
                },
                "stop loss": {
                    "stop_price": stop_loss,
                    "limit_price": stop_price
                },
            }

            return buy

        if self.brokerage == "TD":
            return None  # TODO: Get TD to work so you can write the buy order schema here

    def sell(self, ticker, amount, take_profit=None, stop_loss=None, stop_price=None):

        if self.brokerage == "ALPACA":
            sell = {
                "side": "sell",
                "symbol": ticker,
                "type": "market",
                "qty": amount,
                "time_in_force": "gtc",
                "order_class": "bracket",
                "take_profit": {
                    "limit_price": take_profit
                },
                "stop loss": {
                    "stop_price": stop_loss,
                    "limit_price": stop_price
                },
            }
    # TODO: Add all other advanced orders from alpaca ('OCO', for example, etc.)
