from flask import Flask ,render_template, request      # , send_file
from io import BytesIO
import base64
# import tensorflow as tf
# from tensorflow import keras
# import progressbar
import matplotlib.pyplot as plt
# import math
# import keras
# import pandas as pd
import numpy as np
# from array import array
from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
# from sklearn.model_selection import train_test_split
# from keras.callbacks import EarlyStopping
import yfinance as yf

app = Flask(__name__)


@app.route('/')
def form():
    return render_template('form.html')


@app.route('/data', methods=['POST'])
def hello():
    stock_name = request.form['Name']
    stock = yf.Ticker(stock_name)

    hist = stock.history(period="5y")
    df3 = hist
    d = 30
    n = int(hist.shape[0] * 0.8)
    data = df3.filter(['Close'])
    dataset = data.values

    sc = MinMaxScaler(feature_range=(0, 1))
    scaled_data = sc.fit_transform(dataset)

    train_data = scaled_data[0:n, :]

    X_train = []
    y_train = []
    for i in range(d, len(train_data)):
        X_train.append(train_data[i - d:i, 0])
        y_train.append(train_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(128, return_sequences=False, input_shape=(X_train.shape[1], 1)))
    # model.add(Dropout(0.2))
    # # Adding a second LSTM layer and some Dropout regularisation
    # model.add(LSTM(50, return_sequences=False))
    # model.add(Dropout(0.2))
    # Adding the output layer
    model.add(Dense(25))
    model.add(Dense(1))
    # Compiling the RNN
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Fitting the RNN to the Training set
    history = model.fit(X_train, y_train, epochs=100, batch_size=64)

    test_data = scaled_data[n - d:, :]
    X_test = []
    y_test = dataset[n:, :]
    for i in range(d, len(test_data)):
        X_test.append(test_data[i - d:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predictions = model.predict(X_test)
    predictions = sc.inverse_transform(predictions)

    train = data[:n]
    valid = data[n:]
    valid['predictions'] = predictions
    plt.figure(figsize=(12, 6))
    plt.title('Market Price Prediction')
    plt.xlabel('Time', fontsize=18)
    plt.ylabel('Market Price', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'predictions']])
    plt.legend(['Train', 'Val', 'predictions'], loc='lower right')
    STOCK = BytesIO()
    plt.savefig(STOCK, format="png")

    new_stock = stock
    new_hist = hist

    new_df = hist

    new_df2 = new_df.filter(['Close'])

    last_30_days = new_df2[-d:].values

    last_30_days_scaled = sc.transform(last_30_days)

    x_test = []

    x_test.append(last_30_days_scaled)

    x_test = np.array(x_test)

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions_price = model.predict(x_test)

    predictions_price = sc.inverse_transform(predictions_price)

    STOCK.seek(0)
    plot_url = base64.b64encode(STOCK.getvalue()).decode('utf8')
    return render_template("plot.html", plot_url=plot_url, key=predictions_price)