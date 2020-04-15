import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import dtc


def classification_tree():
    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    clf = clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)
    print(f'First decision tree has {clf.tree_.node_count} nodes with maximum depth {clf.tree_.max_depth}.')

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    # Create a dictionary of all the parameter options
    param_grid = {'criterion': ['gini'],
                  'max_leaf_nodes': list(range(2, 100)),
                  'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                  'min_samples_split': [2, 3, 4, 5]}

    # Create a grid search object
    gsDCT = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, scoring='accuracy')

    # Fit the grid search
    gsDCT.fit(X_train, y_train)

    print(gsDCT.best_estimator_)

    # Predict the response for test dataset
    y_pred = gsDCT.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy after optimization:", metrics.accuracy_score(y_test, y_pred))


def classification_forest():
    # Create a model with 100 trees
    model = RandomForestClassifier(n_estimators=100,
                                   max_features='sqrt',
                                   n_jobs=-1, verbose=1)

    # Fit on training data
    model.fit(X_train, y_train)
    n_nodes = []
    max_depths = []

    for ind_tree in model.estimators_:
        n_nodes.append(ind_tree.tree_.node_count)
        max_depths.append(ind_tree.tree_.max_depth)

    print(f'Average number of nodes {int(np.mean(n_nodes))}')
    print(f'Average maximum depth {int(np.mean(max_depths))}')

    random_pred = model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, random_pred))

    # Optimization for the random forest using random search
    # Hyperparameter grid
    param_grid = {
        'n_estimators': [100, 150, 200, 250],  # The number of trees in the forest.
        'max_depth': [None, 50, 60, 70],  # The maximum depth of the tree.
        'max_features': ['sqrt', None],  # he number of features to consider when looking for the best split
        'min_samples_split': [2, 5, 10],  # The minimum number of samples required to split an internal node
        'bootstrap': [True, False]  # Whether bootstrap samples are used when building trees.
    }

    # Estimator for use in random search
    estimator = RandomForestClassifier()

    # Create the random search model
    rs = RandomizedSearchCV(estimator, param_grid, n_jobs=-1,
                            scoring='accuracy', cv=5,
                            n_iter=1, verbose=1)

    # Fit
    rs.fit(X_train, y_train)
    print(rs.best_params_)

    best_model = rs.best_estimator_
    random_pred = best_model.predict(X_test)
    print("Accuracy after optimization:", metrics.accuracy_score(y_test, random_pred))


def regression_tree():
    rdt = DecisionTreeRegressor(random_state=0)
    print("train data is ", X_train)
    print("test data is", y_train)

    # Train Decision Tree Regression
    rdt = rdt.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = rdt.predict(X_test)
    print(f'First decision tree has {rdt.tree_.node_count} nodes with maximum depth {rdt.tree_.max_depth}.')

    # Model Accuracy, how often is the classifier correct?
    print("MSE:", metrics.mean_squared_error(y_test, y_pred))

    # Create a dictionary of all the parameter options
    param_grid = {
        'max_leaf_nodes': list(range(2, 100)),
        'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        'min_samples_split': [2, 3, 4, 5]}

    # Create a grid search object
    gsDCT = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=5)

    # Fit the grid search
    gsDCT.fit(X_train, y_train)

    print(gsDCT.best_estimator_)
    # Predict the response for test dataset
    y_pred = gsDCT.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    print("MSE after optimization:", metrics.mean_squared_error(y_test, y_pred))


def regression_forest():
    # Create a model with 100 trees
    regression_model = RandomForestRegressor(n_estimators=100,
                                             max_features='sqrt',
                                             n_jobs=-1, verbose=1)

    # Fit on training data
    regression_model.fit(X_train, y_train)
    n_nodes = []
    max_depths = []

    for ind_tree in regression_model.estimators_:
        n_nodes.append(ind_tree.tree_.node_count)
        max_depths.append(ind_tree.tree_.max_depth)

    print(f'Average number of nodes {int(np.mean(n_nodes))}')
    print(f'Average maximum depth {int(np.mean(max_depths))}')

    random_pred = regression_model.predict(X_test)
    print("MSE :", metrics.mean_squared_error(y_test, random_pred))

    # Optimization for the random forest using random search
    # Hyperparameter grid
    param_grid = {
        'n_estimators': [100, 150, 200, 250],  # The number of trees in the forest.
        'max_depth': [None, 50, 60, 70],  # The maximum depth of the tree.
        'max_features': ['sqrt', None],  # he number of features to consider when looking for the best split
        'min_samples_split': [2, 5, 10],  # The minimum number of samples required to split an internal node
        'bootstrap': [True, False]  # Whether bootstrap samples are used when building trees.
    }

    # Estimator for use in random search
    estimator = RandomForestRegressor()

    # Create the random search model
    rs = RandomizedSearchCV(estimator, param_grid, n_jobs=-1,
                            cv=5,
                            n_iter=1, verbose=1)

    # Fit
    rs.fit(X_train, np.ravel(y_train, order='C'))
    print(rs.best_params_)

    best_model = rs.best_estimator_
    random_pred = best_model.predict(X_test)
    print("MSE after optimization:", metrics.mean_squared_error(y_test, random_pred))


if __name__ == "__main__":
    data = pd.read_excel("adult_income.xlsx", sheet_name='adult_income_data', header=0)  # 30162 rows

    # 1)	Train: rows 1-19,034.
    # 2)	Validation: rows 19,035-24,130.
    # 3)	Test: rows 24,130-end.

    feature_cols = data.columns
    print(feature_cols)
    # get all coloumns but the last ( last columns is the class )
    feature_cols = feature_cols[:-1]
    X = data[feature_cols]

    # (1) which is classification using a tree and a forest

    # DATA FOR CLASSIFICATION

    # encode all categorical features
    X = dtc.encode_feaures(X)
    print(data.columns[-1])
    # class is the last columns
    y = data[data.columns[-1]]
    target_col = data.columns[-1]
    X.to_csv("newdata.csv")

    y = y.to_numpy()
    X_2 = X.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X_2, y, test_size=0.19995, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2111, random_state=1)

    # BUILD THE TREE / FOREST

    # classification_tree()
    # classification_forest()

    # (2) Regression using both a tree and forest

    # DATA FOR REGRESSION
    cols_to_drop = ['education', 'education-num', 'occupation', '>50K']
    new_y = X[['education-num']]
    new_X = data.drop(cols_to_drop, axis=1)

    # encode all categorical features
    new_X = dtc.encode_feaures(new_X)
    # class is the last columns
    new_X.to_csv("newnewdata.csv")

    new_y = new_y.to_numpy()
    new_X2 = new_X.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(new_X2, new_y, test_size=0.19995, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2111, random_state=1)

    # BUILD THE TREE / FOREST

    # regression_tree()
    # regression_forest()

