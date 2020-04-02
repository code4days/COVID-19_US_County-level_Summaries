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
from sklearn.model_selection import GridSearchCV
from xgboost import plot_tree
from xgboost import plot_importance
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.tree import DecisionTreeRegressor
from pylab import rcParams
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
import shap
os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/Graphviz2.38/bin/'


def create_confirmed_dict(data_dir):
    filename = join(data_dir, 'deaths_timeseries.csv')
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


def exponentialLoss(y, pred):
    df = -y * np.exp(-y * pred)
    hess = np.exp(-y * pred)
    return df, hess


def grid_search(X_data,Y_data):
    len_X = X_data.shape[1] - Y_data.shape[1]
    Y = X_data.iloc[:, -1]
    X = X_data.iloc[:, 3:len_X]
    print(X.shape, Y.shape, len_X)

    data_dmatrix = xgb.DMatrix(data=X, label=Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=123)
    # Define initial best params and MAE
    params = {
        # Parameters that we are going to tune.
        'max_depth': 3,
        'min_child_weight': 1,
        'eta': .01,
        'subsample': 0.4,
        'colsample_bytree': 0.2,
        'learning_rate': 0.01,
        'objective': 'reg:squaredlogerror'
    }
    min_mae = float("Inf")
    num_boost_round = 100
    gridsearch_params = [
            (max_depth, min_child_weight,subsample, colsample)
            for max_depth in range(1, 5)
            for min_child_weight in range(1, 5)
            for subsample in [i / 10. for i in range(1, 11)]
            for colsample in [i / 10. for i in range(1, 11)]]
    best_params = None

    for max_depth, min_child_weight,subsample, colsample in gridsearch_params:
        print("CV with max_depth={}, min_child_weight={}".format(
            max_depth,
            min_child_weight))
        print("CV with subsample={}, colsample={}".format(
            subsample,
            colsample))

        # Update our parameters
        params['max_depth'] = max_depth
        params['min_child_weight'] = min_child_weight
        params['subsample'] = subsample
        params['colsample_bytree'] = colsample

        # Run CV
        cv_results = xgb.cv(
            params,
            data_dmatrix,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=5,
            metrics={'mae'},
            early_stopping_rounds=150
        )

        # Update best MAE
        mean_mae = cv_results['test-mae-mean'].min()
        boost_rounds = cv_results['test-mae-mean'].argmin()
        print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
        if mean_mae < min_mae:
            min_mae = mean_mae
            best_params = (max_depth, min_child_weight, subsample, colsample)

    print(
        "Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], best_params[2], best_params[3], min_mae))


def run_xgboost(X_data,Y_data):
    len_X = X_data.shape[1] - Y_data.shape[1]
    Y = X_data.iloc[:, -1]
    X = X_data.iloc[:, 3:len_X]
    print(X.shape, Y.shape, len_X)

    data_dmatrix = xgb.DMatrix(data=X, label=Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123)
    dtest = xgb.DMatrix(X_test, label=y_test)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    num_boost_round = 500

    params = {
        # Parameters that we are going to tune.
        'max_depth': 4,
        'min_child_weight': 1,
        'eta': .01,
        'subsample': 0.4,
        'colsample_bytree': 0.2,
        'learning_rate' : 0.01,
        'objective' : 'reg:squaredlogerror'
       # 'objective': 'reg:squarederror',
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtest, "Test")],
        early_stopping_rounds=300
    )
    print("Best MAE: {:.2f} in {} rounds".format(model.best_score, model.best_iteration + 1))

    '''
    xg_reg = xgb.XGBRegressor(objective='reg:squaredlogerror', colsample_bytree=0.2, gamma=0, subsample=0.4,
                              learning_rate=0.01,
                              max_depth=10, alpha=10, n_estimators=200)
    xg_reg.fit(X_train, y_train)

    cv_results = xgb.cv(
        params,
        data_dmatrix,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        metrics={'rmse'},
        early_stopping_rounds=150
    )
    print(cv_results)
    '''

    plt.rcParams['figure.figsize'] = 60, 20
    xgb.plot_tree(model, num_trees=1)
    plt.show()

    # accuracy
    y_pred = model.predict(dtest)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    # rmse
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("RMSE: %f" % (rmse))

    # mae
    mean_train = np.mean(y_train)
    baseline_predictions = np.ones(y_test.shape) * mean_train
    mae_baseline = mean_absolute_error(y_test, baseline_predictions)
    print("Baseline MAE is {:.2f}".format(mae_baseline))

    xgb.plot_importance(model)
    #print(xg_reg.get_booster().get_score(importance_type="gain"))
    print(model.get_score(importance_type="gain"))
    plt.rcParams['figure.figsize'] = [5, 5]
    plt.show()

    # explain the model's predictions using SHAP
    shap_values = shap.TreeExplainer(model).shap_values(X_train)
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    X = X_train
    feat_df = pd.DataFrame(sorted(model.get_fscore().items(),
                                  key=lambda x: x[1],
                                  reverse=True),
                           columns=['Feature',
                                    'Importance'])

    feats = feat_df.sort_values('Importance')[::-1]['Feature'].head(15)

    column_index = []
    new_shap_ar = []
    new_X = []
    for c in feats.values:
        column_index = list(X.columns).index(c)
        new_shap_ar.append(shap_values[:, column_index:column_index + 1])
        new_X.append(X.iloc[:, column_index])

    new_X = pd.concat(new_X, axis=1)
    new_shap_ar = np.hstack(new_shap_ar)

    shap.summary_plot(new_shap_ar, new_X)
    #shap.force_plot(explainer.expected_value, shap_values[10,:], X_test.iloc[10,:])



def main():
    data_dir = r"D:\JHU\corona\disease_spread\data"
    counties_dict = create_counties_dict(data_dir)
    confirmed_dict = create_confirmed_dict(data_dir)
    print(counties_dict.shape, confirmed_dict.shape)

    df = counties_dict.merge(confirmed_dict, on='FIPS', how='inner')
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]

    run_xgboost(df,confirmed_dict)
    #grid_search(df,confirmed_dict)


if __name__ == '__main__':
    main()
