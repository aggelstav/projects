import random
import warnings
from collections import deque
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np

import modin.pandas as pd
import seaborn as sns
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import (accuracy_score, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, KFold, StratifiedKFold,
                                     train_test_split)
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier, plot_importance, plot_tree

sns.set(rc={'figure.figsize': (17, 11)})
sns.set_palette("husl")

warnings.filterwarnings('ignore')


dataset = pd.read_csv(
    "/home/aggelos/Dropbox/Diplomatiki/MObility/mobility_dataset.csv", index_col=0)

look_ahead = 1  # how far into the future are we trying to predict?
LOOKBACK = 1

predict_cell = "5ALTE"  # The cell of interest


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

    X = np.array(X)
    X = X.reshape(X.shape[0], X.shape[2])

    return X, y


""" We need to sample out some data """
df = dataset
times = sorted(df.index.values)
last_5pct = times[-int(0.2*len(times))]

validation_df = df[(df.index >= last_5pct)]
train_df = df[(df.index < last_5pct)]


train_x, train_y = preprocess_df(
    train_df, predict_cell, look_ahead, lookback=LOOKBACK)
test_x, test_y = preprocess_df(
    validation_df, predict_cell, look_ahead, lookback=LOOKBACK)


def objective(space):

    clf = XGBClassifier(
        n_estimators=space['n_estimators'],
        learning_rate=space['learning_rate'],
        max_depth=space['max_depth'],
        min_child_weight=space['min_child_weight'],
        subsample=space['subsample'])

    eval_set = [(train_x, train_y), (test_x, test_y)]

    clf.fit(train_x, train_y,
            eval_set=eval_set, eval_metric="auc",
            early_stopping_rounds=30)

    pred = clf.predict_proba(test_x)[:, 1]
    auc = roc_auc_score(test_y, pred)
    print("SCORE:", auc)

    return{'loss': 1-auc, 'status': STATUS_OK}


space = {
    'n_estimators': hp.choice('n_estimators', np.arange(100, 5000, 200, dtype=int)),
    'learning_rate': hp.choice('learning_rate', [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]),
    'max_depth': hp.choice('max_depth', np.arange(1, 14, dtype=int)),
    'min_child_weight': hp.quniform('x_min_child', 1, 10, 1),
    'subsample': hp.uniform('x_subsample', 0.8, 1)
}


trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)

print(best)
