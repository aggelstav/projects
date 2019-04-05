import math

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import xgboost as xgb
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (GridSearchCV, KFold, StratifiedKFold,
                                     train_test_split)
from sklearn.preprocessing import MinMaxScaler
from xgboost import plot_importance, plot_tree

df = pd.read_csv(
    "/home/aggelos/Dropbox/Diplomatiki/MObility/mobility_dataset.csv", index_col=0)
series = df['3CLTE']

# transform time series into supervised learning format


def series_to_supervised(data, lookback=1, n_out=1):
    df = pd.DataFrame(data)
    cols = list()
    # input sequence (t-n, ..., t-1)
    for i in range(lookback, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = pd.concat(cols, axis=1)
    # drop rows with nan values
    agg.dropna(inplace=True)
    return agg


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


def rolling_evaluation(df, lookback=2, count=1):
    scaler = MinMaxScaler()

    train, test = train_test_split(df.values)

    X_train = train[:, :-1]
    X_train = scaler.fit_transform(X_train)
    # X_train = X_train.reshape((1, lookback))
    y_train = train[:, -1]

    X_test = test[:, :-1]
    X_test = scaler.fit_transform(X_test)
    y_test = test[:, -1]

    predictor = reg.fit(X_train, y_train,
                        eval_set=[(X_train, y_train), (X_test, y_test)],
                        early_stopping_rounds=50,
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


# scaler = MinMaxScaler()
lookback = 32
df = series_to_supervised(series, lookback, 1)
# X = df.values[:, :-1]
# X = scaler.fit_transform(X)
# y = df.values[:, 1]

# reg = xgb.XGBRegressor()
# n_estimators = [100, 200, 300, 400, 500, 1000]
# learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4]
# param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators)
# kfold = KFold(n_splits=10, shuffle=False, random_state=7)
# grid_search = GridSearchCV(
# reg, param_grid, scoring="neg_mean_squared_error", n_jobs=4, cv=kfold)
# grid_result = grid_search.fit(X, y)

# print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
# print("%f (%f) with: %r" % (mean, stdev, param))
# # plot results
# scores = np.array(means).reshape(len(learning_rate), len(n_estimators))
# for i, value in enumerate(learning_rate):
# plt.plot(n_estimators, scores[i], label='learning_rate: ' + str(value))
# plt.legend()
# plt.xlabel('n_estimators')
# plt.ylabel('Log Loss')
# plt.show()

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
