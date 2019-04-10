import random
import warnings
from collections import deque
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import seaborn as sns
import tensorflow as tf
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import (GRU, LSTM, Activation, BatchNormalization,
                                     Conv1D, Dense, Dropout, Flatten,
                                     GlobalAveragePooling1D, Input, Lambda,
                                     MaxPooling1D, Permute, RepeatVector,
                                     Reshape, TimeDistributed, concatenate)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

sns.set(rc={'figure.figsize': (17, 11)})
sns.set_palette("husl")

warnings.filterwarnings('ignore')


dataset = pd.read_csv(
    "sales-of-shampoo-over-a-three-ye.csv", index_col=0)

look_ahead = 1  # how far into the future are we trying to predict?

predict_cell = "4CLTE"  # The cell of interest


def classify(current, future):

    if float(future) > float(current):
        return 1
    else:
        return 0


def preprocess_df(df, predict_cell, look_ahead, lookback):
    scaler = MinMaxScaler()

    df['future'] = df[f"{predict_cell}"].shift(-look_ahead)

    df['target'] = list(map(classify, df[f"{predict_cell}"], df["future"]))

    df.drop('future', 1, inplace=True)
    df[df.columns] = scaler.fit_transform(df[df.columns])
    df.dropna(inplace=True)

    sequential_data = []  # this is a list that contains sequences
    # those will be our actual sequences. They are made with deque
    prev_days = deque(maxlen=lookback)

    for i in df.values:  # iterate over values
        prev_days.append([n for n in i[:-1]])  # store all but the target
        if len(prev_days) == lookback:  # make sure we have the right number of sequences sequences!
            # append the sequences
            sequential_data.append([np.array(prev_days), i[-1]])

    random.shuffle(sequential_data)

    downs = []  # list that will store our buy sequences and targets
    ups = []  # list that will store our sell sequences and targets

    for seq, target in sequential_data:  # iterate over the sequential data
        if target == 0:
            downs.append([seq, target])
        elif target == 1:
            ups.append([seq, target])

    random.shuffle(ups)  # shuffle the buys
    random.shuffle(downs)  # shuffle the sells!

    lower = min(len(ups), len(downs))  # what's the shorter length?

    # make sure both lists are only up to the shortest length.
    ups = ups[:lower]
    # make sure both lists are only up to the shortest length.
    downs = downs[:lower]

    sequential_data = ups+downs  # add them together
    random.shuffle(sequential_data)

    X = []
    y = []
    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), y


""" We need to sample out some data """
df = dataset
times = sorted(df.index.values)
last_5pct = times[-int(0.2*len(times))]

validation_df = df[(df.index >= last_5pct)]
train_df = df[(df.index < last_5pct)]

# Lets build the mlp_model


def fit_mlp(train_x, train_y):

    inputs = Input(shape=(train_x.shape[1:]))

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

    model.compile(loss='mae',
                  optimizer=opt,
                  metrics=['accuracy'])

    history = model.fit(
        train_x, train_y,
        batch_size=32,
        epochs=20,
        validation_split=0.1,
        verbose=0)

    return model, history

# Lets build the model


def fit_gru(train_x, train_y):

    inputs = Input(shape=(train_x.shape[1:]))

    x = GRU(128, return_sequences=True)(inputs)
    x = Dropout(0.5)(x)

    x = GRU(128, return_sequences=True)(x)
    x = Dropout(0.5)(x)

    x = GRU(128)(x)
    x = Dropout(0.5)(x)

    pred = Dense(1)(x)

    model = Model(inputs, pred)

    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

    model.compile(loss='mae',
                  optimizer=opt,
                  metrics=['accuracy'])

    history = model.fit(
        train_x, train_y,
        batch_size=32,
        epochs=20,
        verbose=0,
        validation_split=0.1)

    return model, history


def fit_fcn(train_x, train_y):

    inputs = Input(shape=(train_x.shape[1:]))

    x = Conv1D(64, 4, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(64, 4, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(128, 4, padding='same')(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling1D()(x)

    prediction = Dense(1)(x)

    model = Model(inputs, prediction)

    model.compile(loss='mae',
                  optimizer=tf.keras.optimizers.Adam(lr=0.001, decay=1e-6),
                  metrics=['accuracy'])

    history = model.fit(train_x, train_y,
                        batch_size=32,
                        epochs=20,
                        validation_split=0.1,
                        verbose=0)

    return model, history


def fit_cnn_gru(train_x, train_y):
    inputs = Input(train_x.shape[1:])

    x = Conv1D(16, 2, activation='relu')(inputs)

    x = GRU(64, activation='relu')(x)

    x = Dropout(0.5)(x)

    preds = Dense(1)(x)

    model = Model(inputs, preds)

    model.compile(loss='mae',
                  optimizer=tf.keras.optimizers.Adam(lr=0.001, decay=1e-6),
                  metrics=['accuracy'])

    history = model.fit(train_x, train_y,
                        batch_size=32,
                        epochs=20,
                        validation_split=0.1,
                        verbose=0)

    return model, history


train_x, train_y = preprocess_df(train_df, predict_cell, look_ahead, 16)
test_x, test_y = preprocess_df(validation_df, predict_cell, look_ahead, 16)
cnn_gru, _ = fit_cnn_gru(train_x, train_y)
cnn_gru_preds = cnn_gru.predict(test_x)


train_x, train_y = preprocess_df(train_df, predict_cell, look_ahead, 2)
test_x, test_y = preprocess_df(validation_df, predict_cell, look_ahead, 2)
mlp, _ = fit_mlp(train_x, train_y)
mlp_preds = mlp.predict(test_x)

train_x, train_y = preprocess_df(train_df, predict_cell, look_ahead, 2)
test_x, test_y = preprocess_df(validation_df, predict_cell, look_ahead, 2)
gru, _ = fit_gru(train_x, train_y)
gru_preds = gru.predict(test_x)


train_x, train_y = preprocess_df(train_df, predict_cell, look_ahead, 64)
test_x, test_y = preprocess_df(validation_df, predict_cell, look_ahead, 64)
fcn, _ = fit_fcn(train_x, train_y)
fcn_preds = fcn.predict(test_x)

predictions = (mlp_preds[-210:, :] +
               fcn_preds[-210:, :] + gru_preds[-210:, :] + cnn_gru_preds[-210:, :])/4
predictions = predictions[:, 1].round()
acc_score = accuracy_score(test_y[-50:], predictions[-50:])
print(acc_score)

predictions = pd.DataFrame({'mlp': mlp_preds[-210:, 1],
                            'gru': gru_preds[-210:, 1],
                            'fcn': fcn_preds[-210:, 1],
                            'cnn-gru': cnn_gru_preds[-210:, 1]})

data = pd.DataFrame({'mlp': mlp_preds[-210:, 1],
                     'gru': gru_preds[-210:, 1],
                     'fcn': fcn_preds[-210:, 1],
                     'cnn-gru': cnn_gru_preds[-210:, 1],
                     'actual': test_y[-210:]})


# print(data)

predictions = predictions.values

y = test_y[-210:]

x_test = predictions[-50:, :]
x_train = predictions[:-50, :]

y_test = y[-50:]
y_train = y[:-50]


def nn(train_x, train_y):
    inputs = Input(shape=(4,))
    x = Dense(32, activation='relu')(inputs)
    output = Dense(1)(inputs)

    model = Model(inputs, output)
    model.compile(optimizer='adam', loss='mse')

    stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto', baseline=None)

    history = model.fit(train_x, train_y, verbose=0,
                        epochs=800, validation_split=0.2,
                        callbacks=[stop])

    return model, history


model, _ = nn(x_train, y_train)
final_preds = model.predict(x_test)
final_preds = final_preds.round()
final_acc_score = accuracy_score(y_test, final_preds)
print(f"Final Accuracy: {final_acc_score}")
