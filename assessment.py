def risk(trade_info):  # This function should calculate traditional risk parameters as well as context-based risk (
    # risk that takes into account all other portfolio positions and the risks that those positions carry in addition
    # to the risk of executing the trade passed to the function)

    # Standard-deviation based risk:

    return None


def hedge():  # This should hedge the trade based on the risk calculated above, if needed [algorithm makes a
    # prediction and formulates a potential trade -> gets passed to the above function that calculates
    # the risk of making that trade with a given amount of capital -> additional logic should execute based on the
    # algorithm's percent confidence of the prediction AND the risk calculated above that generates the proper hedge
    # position. (Hedges vary from a simple stop-loss price to a trailing stop-loss/trailing percentage stop-loss to
    # entirely new positions)
    return None


def confirm_trade():
    # This function should confirm that the trade the algorithm is proposing to make checks out with what is allowed
    # (i.e. trade doesn't cost more than the brokerage balance, the trade is submitted with the proper hedge, etc.)

    return None
