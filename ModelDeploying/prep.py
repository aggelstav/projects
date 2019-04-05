import json
import random
from collections import deque
import codecs


import numpy as np
from numpy import argmax

import modin.pandas as pd
from sklearn.preprocessing import MinMaxScaler


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

test_x = test_x.tolist()
json_file = "test.json"
json.dump(test_x, codecs.open(json_file, 'w',
                              encoding='utf-8'), sort_keys=True, indent=4)
