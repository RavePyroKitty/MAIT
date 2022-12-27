def mfi_overbought_indicator(mfi_value, overbought=80):
    if mfi_value > overbought:
        return 1
    else:
        return 0


def mfi_oversold_indicator(mfi_value, oversold=20):
    if mfi_value < oversold:
        return 1
    else:
        return 0
