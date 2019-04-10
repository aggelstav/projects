import random
import warnings
from collections import deque
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow.keras as keras
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import (GRU, LSTM, Activation, BatchNormalization,
                                     Conv1D, Dense, Dropout, Flatten,
                                     GlobalAveragePooling1D, Input, Lambda,
                                     MaxPooling1D, Permute, RepeatVector,
                                     Reshape, TimeDistributed, concatenate)
from tensorflow.keras.models import Model, Sequential

sns.set(rc={'figure.figsize': (17, 11)})
sns.set_palette("husl")

warnings.filterwarnings('ignore')


dataset = pd.read_csv(
    "/home/aggelos/Dropbox/Diplomatiki/MObility/mobility_dataset.csv", index_col=0)

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

# define custom learning rate schedule


class CosineAnnealingLearningRateSchedule(Callback):
        # constructor
    def __init__(self, n_epochs, n_cycles, lrate_max, verbose=0):
        self.epochs = n_epochs
        self.cycles = n_cycles
        self.lr_max = lrate_max
        self.lrates = list()

    # calculate learning rate for an epoch
    def cosine_annealing(self, epoch, n_epochs, n_cycles, lrate_max):
        epochs_per_cycle = np.floor(n_epochs/n_cycles)
        cos_inner = (np.pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
        return lrate_max/2 * (np.cos(cos_inner) + 1)

    # calculate and set learning rate at the start of the epoch
    def on_epoch_begin(self, epoch, logs=None):
        # calculate learning rate
        lr = self.cosine_annealing(
            epoch, self.epochs, self.cycles, self.lr_max)
        # set learning rate
        backend.set_value(self.model.optimizer.lr, lr)
        # log value
        self.lrates.append(lr)


"""We try snapshot ensemble so we need an aggresive learning rate schedule """

""" We need to sample out some data """
df = dataset
times = sorted(df.index.values)
last_5pct = times[-int(0.2*len(times))]

validation_df = df[(df.index >= last_5pct)]
train_df = df[(df.index < last_5pct)]


def fit_gru(train_x, train_y, test_x, test_y, n_epochs, callbacks):

    inputs = Input(shape=(train_x.shape[1:]))

    x = Conv1D(64, 4, activation='relu')(inputs)

    x = GRU(128, dropout=0.2, recurrent_dropout=0.5,
            return_sequences=True)(x)
    x = GRU(128, dropout=0.2, recurrent_dropout=0.5)(x)

    pred = Dense(2, activation='softmax')(x)

    model = Model(inputs, pred)

    opt = keras.optimizers.Adam(lr=0.001, decay=1e-6)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    history = model.fit(
        train_x, train_y,
        batch_size=64,
        epochs=n_epochs,
        verbose=1,
        validation_data=(test_x, test_y))

    return model, pd.DataFrame(history.history)

# define learning rate callback for snapshot ensemble
n_epochs = 200
n_cycles = n_epochs / 25
ca = CosineAnnealingLearningRateSchedule(n_epochs, 8, 0.01)

lookback = 64


train_x, train_y = preprocess_df(train_df, predict_cell, look_ahead, lookback)
test_x, test_y = preprocess_df(
    validation_df, predict_cell, look_ahead, lookback)
gru, gru_history = fit_gru(train_x, train_y, test_x, test_y, n_epochs, ca)
gru_history.plot()
plt.show()
gru_preds = gru.predict(test_x)
