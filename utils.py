import pandas as pd


def normalizer(data):
    """

    :param data: DataFrame of which the values are to be normalized
    :return: DataFrame in the format of the input with values normalized column-wise
    """
    norm = pd.DataFrame()
    cols = list(data)

    scaling_values = dict()
    minimum = 0
    maximum = 0

    for cols in cols:
        c = data[cols]

        try:
            if len(c.unique()) == 1:
                norm[cols] = c.values
                continue
        except Exception as e:
            print('An exception occurred:', e)
            pass

        else:
            try:
                c_vals = c.values
                c = (c_vals - c_vals.min()) / (c_vals.max() - c_vals.min())
                minimum = c_vals.min()
                maximum = c_vals.max()
                scaling_values.update({cols: (minimum, maximum)})
            except Exception:

                pass
        norm[cols] = c

    norm = norm.fillna(value=0)

    return norm, scaling_values


def data_denormalize(data, min, max):
    denormalized = []

    for i in (range(len(data))):
        denormalized.append((data[i] * (max - min)) + min)

    return denormalized
