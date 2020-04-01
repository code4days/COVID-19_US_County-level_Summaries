"""
Try xgboost
"""

from os.path import join, exists
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import csv
import matplotlib.pyplot as plt
import os
from xgboost import XGBClassifier
from xgboost import plot_tree
from xgboost import plot_importance
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.tree import DecisionTreeRegressor
from pylab import rcParams
os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/Graphviz2.38/bin/'


def create_confirmed_dict(data_dir):
    filename = join(data_dir, 'infections_timeseries.csv')
    df = pd.read_csv(filename, dtype={"FIPS": int})
    df['FIPS'] = df['FIPS'].apply(lambda x: str(x).zfill(5))
    return df


def create_counties_dict(data_dir):
    filename = join(data_dir, 'counties.csv')
    df = pd.read_csv(filename, dtype={"FIPS": int})
    df['FIPS'] = df['FIPS'].apply(lambda x: str(x).zfill(5))
    return df


def get_last_date(df):
    col_names = list(df.columns.values)
    len_col = len(col_names)
    last_day_str = col_names[len_col - 1]
    last_day = datetime.strptime(last_day_str, '%Y-%m-%d %H:%M:%S').date()
    return last_day


def run_xgboost(X_data,Y_data):
    len_X = X_data.shape[1] - Y_data.shape[1]
    Y = X_data.iloc[:, -1]
    X = X_data.iloc[:, 3:len_X]
    print(X.shape, Y.shape, len_X)

    data_dmatrix = xgb.DMatrix(data=X, label=Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123)
    xg_reg = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.1,
                              max_depth=5, alpha=10, n_estimators=10)

    xg_reg.fit(X_train, y_train)

    plt.rcParams['figure.figsize'] = 60, 20
    xgb.plot_tree(xg_reg, num_trees=0)
    plt.show()

    preds = xg_reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    xgb.plot_importance(xg_reg)
    print(xg_reg.get_booster().get_score(importance_type="gain"))
    plt.rcParams['figure.figsize'] = [5, 5]
    print("RMSE: %f" % (rmse))
    plt.show()




def main():
    data_dir = r"D:\JHU\corona\disease_spread\data"
    counties_dict = create_counties_dict(data_dir)
    confirmed_dict = create_confirmed_dict(data_dir)
    print(counties_dict.shape, confirmed_dict.shape)

    df = counties_dict.merge(confirmed_dict, on='FIPS', how='inner')
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]

    run_xgboost(df,confirmed_dict)


if __name__ == '__main__':
    main()
