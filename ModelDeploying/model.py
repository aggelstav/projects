import pickle
import random
from collections import deque

import numpy as np
from numpy import argmax

import modin.pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (GRU, LSTM, Activation, BatchNormalization,
                                     Conv1D, Dense, Dropout, Flatten,
                                     GlobalAveragePooling1D, Input,
                                     concatenate)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam


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
    # another shuffle, so the model doesn't get confused with all 1 class then the other.
    random.shuffle(sequential_data)

    X = []
    y = []
    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), y


def fcn(train_x, train_y):

    inputs = Input(shape=(train_x.shape[1:]))

    x = Conv1D(64, 2, activation='relu')(inputs)
    x = BatchNormalization()(x)

    x = Conv1D(64, 2, activation='relu')(x)
    x = BatchNormalization()(x)

    x = Conv1D(64, 2, activation='relu')(x)

    x = GlobalAveragePooling1D()(x)

    pred = Dense(2, activation='softmax')(x)

    model = Model(inputs, pred)

    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

    es = EarlyStopping(monitor='val_loss', mode='min',
                       min_delta=1, patience=20)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    history = model.fit(
        train_x, train_y,
        batch_size=32,
        epochs=EPOCHS,
        validation_split=0.1,
        callbacks=[es],
        verbose=1)

    history = pd.DataFrame(history.history)
    history.plot()

    return model, history

# Lets build the model


def wide(train_x, train_y):

    inputs = Input(shape=(train_x.shape[1:]))

    x = Flatten()(inputs)

    x = Dense(4196, activation='relu')(x)
    x = Dropout(0.91)(x)
    x = Dense(32, activation='relu')(x)

    pred = Dense(2, activation='softmax')(x)

    model = Model(inputs, pred)

    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

    es = EarlyStopping(monitor='val_loss', mode='min',
                       min_delta=1, patience=10)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    history = model.fit(
        train_x, train_y,
        batch_size=64,
        epochs=EPOCHS,
        validation_split=0.1,
        callbacks=[es],
        verbose=0)

    history = pd.DataFrame(history.history)
    history.plot()

    return model, history
# Lets build the model


def mlp(train_x, train_y):

    inputs = Input(shape=(train_x.shape[1:]))

    x = Flatten()(inputs)

    x = Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.001),
              activation='relu')(x)
    x = Dropout(0.54)(x)

    x = Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.001),
              activation='relu')(x)
    x = Dropout(0.33)(x)

    x = Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.001),
              activation='relu')(x)
    x = Dropout(0.10)(x)

    pred = Dense(2, activation='softmax')(x)

    model = Model(inputs, pred)

    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

    es = EarlyStopping(monitor='val_loss', mode='min',
                       min_delta=1, patience=30)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    history = model.fit(
        train_x, train_y,
        batch_size=32,
        epochs=EPOCHS,
        validation_split=0.1,
        callbacks=[es],
        verbose=0)

    history = pd.DataFrame(history.history)
    history.plot()

    return model, history

# Lets build the model


def gru(train_x, train_y):

    inputs = Input(shape=(train_x.shape[1:]))

    x = GRU(128, activation='relu', return_sequences=True)(inputs)
    x = Dropout(0.04)(x)

    x = GRU(128, activation='relu', return_sequences=True)(x)
    x = Dropout(0.11)(x)

    x = GRU(128, activation='relu')(x)
    x = Dropout(0.11)(x)
    x = Dense(32, activation='relu')(x)

    pred = Dense(2, activation='softmax')(x)

    model = Model(inputs, pred)

    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

    es = EarlyStopping(monitor='val_loss', mode='min',
                       patience=20)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    history = model.fit(
        train_x, train_y,
        batch_size=32,
        epochs=EPOCHS,
        verbose=1,
        validation_split=0.1,
        callbacks=[es])

    history = pd.DataFrame(history.history)
    history.plot()

    return model, history

# Lets build the model


def lstm(train_x, train_y):

    inputs = Input(shape=(train_x.shape[1:]))

    x = LSTM(256)(inputs)
    x = Dropout(0.11)(x)

    x = Dense(32, activation='relu')(x)

    pred = Dense(2, activation='softmax')(x)

    model = Model(inputs, pred)

    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', mode='min',
                       min_delta=1, patience=10)

    history = model.fit(
        train_x, train_y,
        batch_size=32,
        epochs=EPOCHS,
        verbose=0,
        validation_split=0.1,
        callbacks=[es])

    history = pd.DataFrame(history.history)
    history.plot()

    return model, history

# define stacked model from multiple member input models


def define_stacked_model(members):
    # update all layers in all models to not be trainable
    for i in range(len(members)):
        model = members[i]
        for layer in model.layers:
            # print(layer.name)
            # make not trainable
            layer.trainable = False
            # rename to avoid 'unique layer name' issue
            #layer.name = 'ensemble_' + str(i+1)
    # define multi-headed input
    ensemble_visible = [model.input for model in members]
    # concatenate merge output from each model
    ensemble_outputs = [model.output for model in members]
    merge = concatenate(ensemble_outputs)

    hidden = Dense(1024, activation='relu')(merge)
    output = Dense(2, activation='softmax')(hidden)
    model = Model(inputs=ensemble_visible, outputs=output)
    # plot graph of ensemble
    #plot_model(model, show_shapes=True)
    # compile
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model

# fit a stacked model


def fit_stacked_model(model, inputX, inputy):
    # prepare input data
    X = [inputX for _ in range(len(model.input))]

    # fit model
    model.fit(X, inputy, epochs=50, verbose=1)


def predict_stacked_model(model, inputX):
    # prepare input data
    X = [inputX for _ in range(len(model.input))]
    # make prediction
    return model.predict(X, verbose=0)


EPOCHS = 50

lookback = 64  # looking back window of 128 timesteps where each timestep
# is 15 minutes

look_ahead = 2  # how far into the future are we trying to predict?

predict_cell = "4CLTE"  # The cell of interest

""" We need to sample out some data """
df = pd.read_csv(
    "/home/aggelos/Dropbox/Diplomatiki/MObility/mobility_dataset.csv", index_col=0)

times = sorted(df.index.values)
last_5pct = times[-int(0.2*len(times))]
last_20pct = times[-int(0.4*len(times))]


test_df = df[(df.index >= last_5pct)]
validation_df = df[((df.index >= last_20pct) & (df.index < last_5pct))]
train_df = df[(df.index < last_20pct)]

"""We need to create our series"""
train_x, train_y = preprocess_df(train_df, predict_cell, look_ahead, lookback)
val_x, val_y = preprocess_df(validation_df, predict_cell, look_ahead, lookback)
test_x, test_y = preprocess_df(test_df, predict_cell, look_ahead, lookback)


mlp_model, mlp_history = mlp(train_x, train_y)
gru_model, gru_history = gru(train_x, train_y)
lstm_model, lstm_history = lstm(train_x, train_y)
wide_model, wide_history = wide(train_x, train_y)
fcn_model, fcn_history = fcn(train_x, train_y)

members = [mlp_model, gru_model, lstm_model, wide_model, fcn_model]

n_members = len(members)

# define ensemble model
stacked_model = define_stacked_model(members)
# fit stacked model on test dataset
fit_stacked_model(stacked_model, val_x, val_y)


# Saving model to disk
stacked_model.save("predictor.h5")


# make predictions and evaluate
yhat = predict_stacked_model(stacked_model, test_x)
yhat = argmax(yhat, axis=1)
acc = accuracy_score(test_y, yhat)
print('Stacked Test Accuracy: %.3f' % acc)
