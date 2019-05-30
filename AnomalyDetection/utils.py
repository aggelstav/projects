import pandas as pd
from pathlib import Path


def feature_selection(features, pathlist, window=10):
    dataframe = pd.DataFrame()
    for feature in features:
        print(1)
        df = mean_rolling_feature(feature, pathlist, window)
        dataframe[feature] = df
    return dataframe


def mean_rolling_feature(feature, pathlist, window):
    df_feature = pd.DataFrame()
    for path in pathlist:
        print(path)
        path_in_str = str(path)
        print(type(feature))
        df = pd.read_csv(path_in_str, delimiter='t')
        df = df[[feature]][:1000].transpose()

        if df_feature.empty:
            df_feature = df
        else:
            df_feature = df_feature.append(df)

    df_feature = df_feature.mean(axis=0)
    df_feature = df_feature.rolling(window=window).mean()
    df_feature.name = str(feature)
    return df_feature

features_selected = ['voltage [V]', 'acceleration (actual) [m/(s*s)]']
pathlist = Path("/home/aggelos-i3/Downloads/simu Elbas/7h33NO").glob(
    '**/*.xls')

df = feature_selection(features=features_selected, pathlist=pathlist)

df.to_csv("normal_dataset.csv")

