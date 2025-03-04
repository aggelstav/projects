import random
import warnings
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import seaborn as sns
#from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import (accuracy_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import (GridSearchCV, KFold, StratifiedKFold,
                                     train_test_split)
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

sns.set(rc={'figure.figsize': (17, 11)})
sns.set_palette("husl")

warnings.filterwarnings('ignore')

dataset = pd.read_csv("dataset.csv", index_col=0)

look_ahead = 1  # how far into the future are we trying to predict?
LOOKBACK = 2

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
        if len(
                prev_days
        ) == lookback:  # make sure we have the right number of sequences sequences!
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

    sequential_data = ups + downs  # add them together
    random.shuffle(sequential_data)

    X = []
    y = []
    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    X = np.array(X)
    #X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

    return X, y


def optimize_classifier(model, X, y):
    n_estimators = [100, 200, 300, 400, 500, 1000]
    learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4]
    max_depth = [3, 4, 6, 8, 10, 12, 14]
    min_child_weight = [1, 3, 5, 7]
    gamma = [0.0, 0.1, 0.2, 0.3, 0.4]
    colsample_bytree = [0.3, 0.4, 0.5, 0.7]
    param_grid = dict(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        gamma=gamma,
        colsample_bytree=colsample_bytree)
    kfold = KFold(n_splits=10, shuffle=False, random_state=7)
    grid_search = GridSearchCV(
        model, param_grid, scoring="neg_log_loss", n_jobs=4, cv=kfold)
    grid_result = grid_search.fit(X, y)

    print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
        # plot results
    scores = np.array(means).reshape(len(learning_rate), len(n_estimators))
    for i, value in enumerate(learning_rate):
        plt.plot(n_estimators, scores[i], label='learning_rate: ' + str(value))
    plt.legend()
    plt.xlabel('n_estimators')
    plt.ylabel('Log Loss')
    plt.show()


dataset = pd.read_csv("dataset.csv", index_col=0)

look_ahead = 1  # how far into the future are we trying to predict?
LOOKBACK = 1

predict_cell = "4CLTE"  # The cell of interest

""" We need to sample out some data """
df = dataset
times = sorted(df.index.values)
last_5pct = times[-int(0.2 * len(times))]

validation_df = df[(df.index >= last_5pct)]
train_df = df[(df.index < last_5pct)]

for predict_cell in list(dataset):
    mean_accuracy = []
    for i in range(1):
        predict_cell = "4CLTE"
        train_x, train_y = preprocess_df(
            train_df, predict_cell, look_ahead, lookback=LOOKBACK)
        test_x, test_y = preprocess_df(
            validation_df, predict_cell, look_ahead, lookback=LOOKBACK)
        
        n_input = train_x.shape[1] * train_x.shape[2]
        train_x = train_x.reshape(train_x.shape[0], n_input)
        n_input = test_x.shape[1] * test_x.shape[2]
        test_x = test_x.reshape(test_x.shape[0], n_input)
    
        # fit model
        #model = XGBClassifier()
        #optimize_classifier(model, train_x, train_y)
        model = XGBClassifier(
            learning_rate=0.3,
            n_estimators=3700,
            min_child_weight=2.0,
            max_depth=2,
            subsample=0.85)
        eval_set = [(test_x, test_y)]
        model.fit(
            train_x,
            train_y,
            early_stopping_rounds=10,
            eval_metric="logloss",
            eval_set=eval_set,
            verbose=False)
        # make predictions for test data
        y_pred = model.predict(test_x)
        predictions = [round(value) for value in y_pred]
        # evaluate predictions
        accuracy = accuracy_score(test_y, predictions)
        mean_accuracy.append(accuracy)
    avg_mean_accuracy = np.mean(mean_accuracy)
    print(mean_accuracy)
    print(f"For cell:{predict_cell}:")
    print("Accuracy: %.2f%%" % (avg_mean_accuracy * 100.0))
    # ax = plot_importance(model)
    plt.boxplot(mean_accuracy)

