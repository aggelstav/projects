import pandas as pd

def feature_selection(features, pathlist, window=10):
    dataframe = []
    for i in range(len(features)):
        f = features[i]
        print(f)
        df = mean_rolling_feature(feature=f, pathlist=pathlist, window=window)
        dataframe.append(df)
    return dataframe


def mean_rolling_feature(feature, pathlist, window):
    df_new = pd.DataFrame()
    for path in pathlist:
        path_in_str = str(path)
        df = pd.read_csv(path_in_str, delimiter='\t')
        column = df.loc[:, feature][:1000]
        if df_new.empty:
            df_new = df
        else:
            df_new = df_new.append(column)
            
    df_new = df_new.mean(axis=1)
    df_feature = df_new.rolling(window=window).mean()
    return df_feature


