import datetime

import keras
import numpy as np
import tensorflow as tf
from keras.layers import ConvLSTM1D, Dense, Resizing
from sklearn.preprocessing import MinMaxScaler

from technical_analysis.technical_classes import TechnicalClasses


class TechnicalTicker(TechnicalClasses):
    def __init__(self, vector_dimensions=64, oscillatory_window=14, volatility_window=14, moving_average_window=28, index_overbought=80, index_oversold=20, ticker='SPY',
                 start_date=None,
                 end_date=datetime.datetime.now(), interval='15m', period='5d'):  # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        super().__init__(oscillatory_window=oscillatory_window, volatility_window=volatility_window, moving_average_window=moving_average_window, index_overbought=index_overbought,
                         index_oversold=index_oversold, ticker=ticker, start_date=start_date, end_date=end_date, interval=interval, period=period)
        self.technical_features_normalized, self.technical_features_normalizer = self.normalize()
        self.technical_classes_encoded, self.technical_class_encoder = self.encode()
        self.vector_dimensions = vector_dimensions

    def normalize(self):
        normalizer = MinMaxScaler()
        normalized = normalizer.fit_transform(X=self.technical_features)

        return normalized, normalizer

    def encode(self):
        encoder = MinMaxScaler()
        encoded = encoder.fit_transform(X=self.technical_classes)  # TODO: Combine this with the above normalize method since I was wrong abt the onehotencode

        # encoder = MultiLabelBinarizer()
        # encoded = encoder.fit_transform(y=self.technical_classes)

        return encoded, encoder

    def shape(self):
        feature_shape = (self.technical_features_normalized.shape[0], self.technical_features_normalized.shape[1])
        class_shape = (self.technical_classes_encoded.shape[0], self.technical_classes_encoded.shape[1])

        return feature_shape, class_shape


class TechnicalThinker:
    def __init__(self, timescales=None, max_batch_size=32, embedding_dimension=64):
        if timescales is None:
            self.timescales = ['1m', '5m', '15m']

        self.timescale_values = {'1m': 1, '2m': 2, '5m': 5, '15m': 15, '30m': 30, '60m': 60, '90m': 90, '1h': 60, '1d': 1440, '5d': 7200, '1wk': 7200, '1mo': 43200, '3mo': 129600}

        self.embedding_dimension = embedding_dimension
        self.batch_size = max_batch_size

    def convolution(self, datasets, epochs=100):

        x_timescale_a = datasets[0][0]
        x_timescale_b = datasets[1][0]
        y_timescale_b = datasets[1][1]

        batch_size_a = datasets[0][2]
        batch_size_b = datasets[1][2]
        print(f'Batch size a: {batch_size_a}, batch size b: {batch_size_b}')

        x_timescale_a = np.asarray(list(x_timescale_a.as_numpy_iterator())).astype('float32')
        x_timescale_b = np.asarray(list(x_timescale_b.as_numpy_iterator())).astype('float32')
        y_timescale_b = np.asarray(list(y_timescale_b.as_numpy_iterator())).astype('float32')

        x_timescale_a = np.nan_to_num(x_timescale_a)
        x_timescale_b = np.nan_to_num(x_timescale_b)
        y_timescale_b = np.nan_to_num(y_timescale_b)

        x_timescale_b = x_timescale_b[:x_timescale_a.shape[0]]  # Ensures there are the same number of samples
        y_timescale_b = y_timescale_b[:x_timescale_a.shape[0]]

        print('x_timescale_a shape pre reshape:', x_timescale_a.shape, x_timescale_a)
        print('x_timescale_b shape pre reshape:', x_timescale_b.shape, x_timescale_b)
        print('y_timescale_b shape pre reshape:', y_timescale_b.shape, y_timescale_b)

        # x_timescale_a = tf.reshape(x_timescale_a, shape=(x_timescale_a.shape[2], x_timescale_a.shape[1] * x_timescale_a.shape[0], 1))  # Shape: (num_features, total_samples,
        # channels [aka num_timescales
        # included in data, so 1])
        # x_timescale_b = tf.reshape(x_timescale_b, shape=(x_timescale_b.shape[2], x_timescale_b.shape[1] * x_timescale_b.shape[0], 1))

        x_timescale_a = tf.reshape(x_timescale_a, shape=(x_timescale_a.shape[0], x_timescale_a.shape[2], x_timescale_a.shape[1], 1))
        x_timescale_b = tf.reshape(x_timescale_b, shape=(x_timescale_b.shape[0], x_timescale_b.shape[2], x_timescale_b.shape[1], 1))
        y_timescale_b = tf.reshape(y_timescale_b, shape=(y_timescale_b.shape[0], y_timescale_b.shape[2], y_timescale_b.shape[1], 1))

        print(f'X_timescale_a shape:, {x_timescale_a.shape}')
        print(f'X_timescale_b shape:, {x_timescale_b.shape}')
        print(f'Y_timescale_b shape, {y_timescale_b.shape}')

        input_shape = x_timescale_a.shape[1:]  # (batch_size [timescale in minutes], width [num_features
        # per batch], height [samples_per_batch], channels)
        output_shape = y_timescale_b.shape[1:]

        filters_conv_a = int(x_timescale_a.shape[1] * x_timescale_a.shape[2])  # TODO: Move all of the reshaping stuff into another function
        filters_conv_b = filters_conv_a // 3
        filters_conv_c = y_timescale_b.shape[1] * y_timescale_b.shape[2] * y_timescale_b.shape[3]  # So that reshape is compatible with the number of values in the array
        filters_conv_d = filters_conv_c // 3
        filters_conv_e = filters_conv_d // 3
        filters_conv_f = 1
        num_features = x_timescale_a.shape[1]
        units = 1000

        kernel = batch_size_b
        pool = batch_size_a // batch_size_b

        print('Conv input shape:', input_shape)
        print('Model output shape:', output_shape)
        print('Filters conv 1:', filters_conv_b)
        print('Pool size:', pool)

        # TODO: Ensure all hyper parameters are appropriate / make this model better / rename model to be cross_timescale_convolution
        cross_layer_convolution = keras.Sequential()
        cross_layer_convolution.add(ConvLSTM1D(kernel_size=kernel, filters=filters_conv_a, activation='relu', input_shape=input_shape, return_sequences=True,
                                               data_format='channels_last'))
        cross_layer_convolution.add(ConvLSTM1D(kernel_size=kernel, filters=filters_conv_b, activation='relu', return_sequences=True, data_format='channels_last'))
        cross_layer_convolution.add(ConvLSTM1D(kernel_size=kernel, filters=filters_conv_c, activation='relu', return_sequences=True, data_format='channels_last'))
        cross_layer_convolution.add(Resizing(height=y_timescale_b.shape[1], width=y_timescale_b.shape[2]))
        cross_layer_convolution.add(Dense(units=num_features, activation='tanh'))
        cross_layer_convolution.add(Dense(units=1, activation='relu'))
        cross_layer_convolution.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        cross_layer_convolution.summary()

        history = cross_layer_convolution.fit(x=x_timescale_a, y=y_timescale_b, batch_size=batch_size_a, epochs=epochs)
        loss = history.history['loss']

        return cross_layer_convolution, loss

    def think(self):
        # TODO: Put all of the data reshaping shit in a separate function
        datasets = []
        # THE GOAL IS TO GET 1m TO PREDICT 5m THEN CLASSIFY BASED ON 5m CLASSIFICATION THEN PREDICT 15m THEN CLASSIFY BASED ON 15m CLASSIFICATION, etc.
        largest_timescale_value = self.timescale_values.get(self.timescales[-1])

        for timescale in self.timescales:
            data_per_timescale = TechnicalTicker(interval=timescale)
            x = data_per_timescale.technical_features_normalized
            y = data_per_timescale.technical_classes_encoded
            print('Y:', y)
            batch_size = (largest_timescale_value // self.timescale_values.get(timescale))
            # Largest timescale should have a batch_size of 1 and then the others are lesser multiples of the largest timescale

            x = tf.data.Dataset.from_tensor_slices(x).batch(batch_size=batch_size, drop_remainder=True)
            y = tf.data.Dataset.from_tensor_slices(y).batch(batch_size=batch_size, drop_remainder=True)

            # print(f"{timescale}: batch_size: {batch_size}, x shape: {x}, x: {list(x.as_numpy_iterator())} y: {list(y.as_numpy_iterator())}")
            # print("batch transposed:", tf.transpose(list(y.as_numpy_iterator())))
            datasets.append([x, y, batch_size])  # [ [x_n1, y_n1, batch_size_n1], [x_n2, y_n2, batch_size_n2], etc. ]

        model = self.convolution(datasets=datasets)

        # TODO: The current cross-timescale embedder outputs raw data about how the market moves (see notes), take the next step
        return None


print(TechnicalThinker().think())
