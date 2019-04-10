import math

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (GridSearchCV, KFold, StratifiedKFold,
                                     train_test_split)
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

df = pd.read_csv(
    "/home/aggelos/projects/MobilityModeling/dataset.csv", index_col=0)
series = df['3CLTE']

# transform time series into supervised learning format


def series_to_supervised(sequence, lookback, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + lookback
        if end_ix > len(sequence) - 1:
            break

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def evaluate(predictions, actual):
    mse = mean_squared_error(actual, predictions)
    mae = mean_absolute_error(actual, predictions)
    mape = mean_absolute_percentage_error(actual, predictions)
    rmse = math.sqrt(mse)
    return rmse, mae, mape


def rolling_evaluation(model, df, lookback=2, count=1):
    scaler = MinMaxScaler()

    train, test = train_test_split(df.values)

    X_train = train[:, :-1]
    X_train = scaler.fit_transform(X_train)
    # X_train = X_train.reshape((1, lookback))
    y_train = train[:, -1]

    X_test = test[:, :-1]
    X_test = scaler.fit_transform(X_test)
    y_test = test[:, -1]

    X_train = series_to_supervised(X_train, lookback, 1)
    X_test = series_to_supervised(X_test, lookback, 1)

    model.fit(
        X_train, y_train,
        verbose=False)  # Change verbose to True if you want to see it train

    predictions = list()
    actual = list()
    for i in range(len(test)):
        x_input = X_test[i, :]
        x_input = x_input.reshape((1, lookback))
        yhat = predictor.predict(x_input)
        predictions.append(yhat[0])
        actual.append(test[i, -1:])

    if count == 1:
        # plot predictions and expected results
        plt.plot(y_train[800:])
        plt.plot([None for i in y_train[800:]] + [x for x in y_test])
        plt.plot([None for i in y_train[800:]] + [x for x in predictions])
        plt.xlabel('Time')
        plt.ylabel('DataRate (MB)')
        plt.title('DataRate Over Time')
        plt.legend(['Train_y', 'Actual', 'Prediction'])
        plt.show()

    rmse, mae, mape = evaluate(predictions, actual)

    return rmse, mae, mape


def fit_mlp(X_train, y_train):

    inputs = Input(shape=(X_train.shape[1:]))

    x = Flatten()(inputs)

    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)

    pred = Dense(1)(x)

    model = Model(inputs, pred)

    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

    model.compile(loss='mse', optimizer=opt)

    history = model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=20,
        validation_split=0.1,
        verbose=0)

    return model, history

    # Lets build the model


def fit_gru(X_train, y_train):

    inputs = Input(shape=(X_train.shape[1:]))

    x = GRU(128, return_sequences=True)(inputs)
    x = Dropout(0.5)(x)

    x = GRU(128, return_sequences=True)(x)
    x = Dropout(0.5)(x)

    x = GRU(128)(x)
    x = Dropout(0.5)(x)

    pred = Dense(1)(x)

    model = Model(inputs, pred)

    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

    model.compile(loss='mse', optimizer=opt)

    history = model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=20,
        verbose=0,
        validation_split=0.1)

    return model, history


def fit_fcn(X_train, y_train):

    inputs = Input(shape=(X_train.shape[1:]))

    x = Conv1D(64, 4, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(64, 4, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(64, 4, padding='same')(x)
    #x = Activation('relu')(x)

    x = GlobalMaxPooling1D()(x)

    prediction = Dense(1)(x)

    model = Model(inputs, prediction)

    model.compile(
        loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.001, decay=1e-6))

    history = model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=20,
        validation_split=0.1,
        verbose=0)

    return model, history


def fit_cnn_gru(X_train, y_train):
    inputs = Input(X_train.shape[1:])

    x = Conv1D(16, 2, activation='relu')(inputs)

    x = GRU(64, activation='relu')(x)

    x = Dropout(0.5)(x)

    preds = Dense(1)(x)

    model = Model(inputs, preds)

    model.compile(
        loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.001, decay=1e-6))

    history = model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=20,
        validation_split=0.1,
        verbose=0)

    return model, history


lookback = 32
scaler = MinMaxScaler()

train, test = train_test_split(df.values)

X_train = train[:, :-1]
X_train = scaler.fit_transform(X_train)
# X_train = X_train.reshape((1, lookback))
y_train = train[-797:, -1]

X_test = test[:, :-1]
X_test = scaler.fit_transform(X_test)
y_test = test[:, -1]

X_train = series_to_supervised(X_train, lookback, 1)
print(X_train.shape)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = series_to_supervised(X_test, lookback, 1)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print(X_train.shape)

cnn_gru, _ = fit_cnn_gru(X_train, y_train)
cnn_gru_preds = cnn_gru.predict(X_test)

mlp, _ = fit_mlp(X_train, y_train)
mlp_preds = mlp.predict(X_test)

gru, _ = fit_gru(X_train, y_train)
gru_preds = gru.predict(X_test)

fcn, _ = fit_fcn(X_train, y_train)
fcn_preds = fcn.predict(X_test)

predictions = (mlp_preds[-210:, :] + fcn_preds[-210:, :] + gru_preds[-210:, :]
               + cnn_gru_preds[-210:, :]) / 4
predictions = pd.DataFrame({
    'mlp': mlp_preds[-210:, 1],
    'gru': gru_preds[-210:, 1],
    'fcn': fcn_preds[-210:, 1],
    'cnn-gru': cnn_gru_preds[-210:, 1]
})
data = pd.DataFrame({
    'mlp': mlp_preds[-210:, 1],
    'gru': gru_preds[-210:, 1],
    'fcn': fcn_preds[-210:, 1],
    'cnn-gru': cnn_gru_preds[-210:, 1],
    'actual': y_test[-210:]
})  # print(data)
predictions = predictions.values

y = y_test[-210:]

X_test = predictions[-50:, :]
x_train = predictions[:-50, :]

y_test = y[-50:]
y_train = y[:-50]


def nn(X_train, y_train):
    inputs = Input(shape=(4, ))
    x = Dense(64, activation='linear')(inputs)
    output = Dense(1)(x)

    model = Model(inputs, output)
    model.compile(optimizer='adam', loss='mse')

    stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=30,
        verbose=0,
        mode='auto',
        baseline=None)

    history = model.fit(
        X_train,
        y_train,
        verbose=0,
        epochs=800,
        validation_split=0.2,
        callbacks=[stop])

    return model, history


df = series_to_supervised(series, lookback, 1)

# repeat experiment
repeats = 30
error_scores = list()
count = 1
for r in range(repeats):
    # walk-forward validation on the test data
    reg = xgb.XGBRegressor(learning_rate=0.1, n_estimator=1000)
    rmse, mae, mape = rolling_evaluation(df, lookback=lookback, count=count)
    error_scores.append(rmse)
    count += 1

# summarize results
results = pd.DataFrame()
results['rmse'] = error_scores
print(results.describe())
results.boxplot()

ax = plot_importance(reg)
ax.plot()
plt.show()
