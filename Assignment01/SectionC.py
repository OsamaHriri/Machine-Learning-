import time
import numpy as np
import pandas as pd
from sklearn import metrics, preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def classification_tree(X_train, X_test, y_train, y_test):
    print("------------------ C - 1 - Classification Tree ------------------")

    model_start_time = time.time()

    # Create a dictionary of all the parameter options
    param_grid = {'criterion': ['gini'],
                  'max_leaf_nodes': list(range(2, 100)),
                  'max_depth': [13, 14, 15, 16],
                  'min_samples_split': [2, 3, 4, 5]}

    # Create a grid search object
    gsDCT = GridSearchCV(DecisionTreeClassifier(), param_grid,
                         cv=5, scoring='accuracy')

    # Fit the grid search
    gsDCT.fit(X_train, y_train)

    print(gsDCT.best_estimator_)

    # Predict the response for test dataset
    y_pred = gsDCT.predict(X_test)

    elapsed_time = time.time() - model_start_time
    print('Elapsed Time : {}', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy :", metrics.accuracy_score(y_test, y_pred))


def classification_forest(X_train, X_test, y_train, y_test):
    print("------------------ C - 1 - Classification Forest ------------------")
    model_start_time = time.time()
    # Hyperparameter grid
    param_grid = {
        'n_estimators': [80, 100, 150, 180],  # The number of trees in the forest.
        'max_depth': [10, 20, 30],  # The maximum depth of the tree.
        'max_features': ['sqrt'],  # he number of features to consider when looking for the best split
        'min_samples_split': [2, 5, 9, 10],  # The minimum number of samples required to split an internal node
        'bootstrap': [True]  # Whether bootstrap samples are used when building trees.
    }

    # Create a grid search object
    gsRFC = GridSearchCV(RandomForestClassifier(), param_grid, n_jobs=-1,
                         scoring='accuracy', cv=5)

    # Fit
    gsRFC.fit(X_train, y_train)
    print(gsRFC.best_params_)

    best_model = gsRFC.best_estimator_
    random_pred = best_model.predict(X_test)

    elapsed_time = time.time() - model_start_time
    print('Elapsed Time : {}', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    print("Accuracy:", metrics.accuracy_score(y_test, random_pred))


def regression_tree(X_train, X_test, y_train, y_test):
    print("------------------ C - 2 - Regression Tree ------------------")
    model_start_time = time.time()
    # Create a dictionary of all the parameter options
    param_grid = {
        'max_leaf_nodes': list(range(10, 60)),
        'max_depth': [10, 12, 14],
        'min_samples_split': [2, 3, 4, 5]}

    # Create a grid search object
    gsDCT = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=5)

    # Fit the grid search
    gsDCT.fit(X_train, y_train)

    print(gsDCT.best_estimator_)
    # Predict the response for test dataset
    y_pred = gsDCT.predict(X_test)

    elapsed_time = time.time() - model_start_time
    print('Elapsed Time : {}', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    # Model Accuracy, how often is the classifier correct?
    print("MSE :", metrics.mean_squared_error(y_test, y_pred))


def regression_forest(X_train, X_test, y_train, y_test):
    print("------------------ C - 2 - Regression Forest ------------------")
    model_start_time = time.time()
    # Hyperparameter grid
    param_grid = {
        'n_estimators': [100, 150, 200, 250],  # The number of trees in the forest.
        'max_depth': [None, 10, 20, 30],  # The maximum depth of the tree.
        'max_features': ['sqrt'],  # he number of features to consider when looking for the best split
        'min_samples_split': [2, 5, 10, 15],  # The minimum number of samples required to split an internal node
        'bootstrap': [True]  # Whether bootstrap samples are used when building trees.
    }

    # Create a grid search object
    gsRFC = GridSearchCV(RandomForestRegressor(), param_grid, n_jobs=-1, cv=5)

    # Fit
    gsRFC.fit(X_train, np.ravel(y_train, order='C'))
    print(gsRFC.best_params_)

    best_model = gsRFC.best_estimator_
    random_pred = best_model.predict(X_test)

    elapsed_time = time.time() - model_start_time
    print('Elapsed Time : {}', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    print("MSE after optimization:", metrics.mean_squared_error(y_test, random_pred))


def encode_features(my_data):
    encode = preprocessing.LabelEncoder()
    features = categorical_cols(my_data)
    for x in features:
        my_data[x] = (encode.fit_transform(my_data[x]))
    return my_data


def categorical_cols(my_data):
    cols = my_data.columns
    num_cols = my_data._get_numeric_data().columns
    feature_col = list(set(cols) - set(num_cols))
    return feature_col


if __name__ == "__main__":
    data = pd.read_excel("adult_income.xlsx", sheet_name='adult_income_data', header=0)  # 30162 rows

    # 1)	Train: rows 1-19,034.
    # 2)	Validation: rows 19,035-24,130.
    # 3)	Test: rows 24,130-end.

    feature_cols = data.columns
    # get all columns but the last ( last columns is the class )
    feature_cols = feature_cols[:-1]
    X = data[feature_cols]

    # (1) which is classification using a tree and a forest

    # DATA FOR CLASSIFICATION

    # encode all categorical features
    X = encode_features(X)

    # class is the last columns
    y = data[data.columns[-1]]
    target_col = data.columns[-1]

    y = y.to_numpy()
    X_2 = X.to_numpy()

    X1_train, X1_test, y1_train, y1_test = train_test_split(X_2, y, test_size=0.19995, random_state=1)

    # BUILD THE TREE / FOREST

    classification_tree(X1_train, X1_test, y1_train, y1_test)
    classification_forest(X1_train, X1_test, y1_train, y1_test)

    # (2) Regression using both a tree and forest

    # DATA FOR REGRESSION
    cols_to_drop = ['education', 'education-num', 'occupation', '>50K']
    y2 = X[['education-num']]
    X2 = data.drop(cols_to_drop, axis=1)

    # encode all categorical features
    X2 = encode_features(X2)

    y2 = y2.to_numpy()
    X2 = X2.to_numpy()

    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.19995, random_state=1)

    # BUILD THE TREE / FOREST

    regression_tree(X2_train, X2_test, y2_train, y2_test)
    regression_forest(X2_train, X2_test, y2_train, y2_test)
