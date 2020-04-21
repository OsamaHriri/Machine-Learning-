import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from Ex1 import encode_feaures


def classification_tree(X_train, X_test, y_train, y_test):
    print("------------------ C - 1 - Classification Tree ------------------")
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
                  'max_depth': [13, 14, 15, 16, 17],
                  'min_samples_split': [2, 3, 4, 5]}

    # Create a grid search object
    gsDCT = GridSearchCV(DecisionTreeClassifier(), param_grid,
                         cv=5, scoring='accuracy')

    # Fit the grid search
    gsDCT.fit(X_train, y_train)

    print(gsDCT.best_estimator_)

    # Predict the response for test dataset
    y_pred = gsDCT.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy after optimization:", metrics.accuracy_score(y_test, y_pred))


def classification_forest(X_train, X_test, y_train, y_test):
    print("------------------ C - 1 - Classification Forest ------------------")
    # Create a model with 100 trees
    rfc = RandomForestClassifier(n_estimators=100, max_features='sqrt',
                                 n_jobs=-1, verbose=1)

    # Fit on training data
    rfc.fit(X_train, y_train)
    n_nodes = []
    max_depths = []

    for ind_tree in rfc.estimators_:
        n_nodes.append(ind_tree.tree_.node_count)
        max_depths.append(ind_tree.tree_.max_depth)

    print(f'Average number of nodes {int(np.mean(n_nodes))}')
    print(f'Average maximum depth {int(np.mean(max_depths))}')

    random_pred = rfc.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, random_pred))

    # Optimization for the random forest using random search
    # Hyperparameter grid
    param_grid = {
        'n_estimators': [100, 150, 200, 250],  # The number of trees in the forest.
        'max_depth': [None, 10, 20, 30],  # The maximum depth of the tree.
        'max_features': ['sqrt'],  # he number of features to consider when looking for the best split
        'min_samples_split': [2, 5, 10, 15],  # The minimum number of samples required to split an internal node
        'bootstrap': [True]  # Whether bootstrap samples are used when building trees.
    }

    # Create a grid search object
    gsRFC = GridSearchCV(RandomForestClassifier(), param_grid, n_jobs=-1,
                         scoring='accuracy', cv=5, verbose=1)

    # Fit
    gsRFC.fit(X_train, y_train)
    print(gsRFC.best_params_)

    best_model = gsRFC.best_estimator_
    random_pred = best_model.predict(X_test)
    print("Accuracy after optimization:", metrics.accuracy_score(y_test, random_pred))


def regression_tree(X_train, X_test, y_train, y_test):
    print("------------------ C - 2 - Regression Tree ------------------")
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
        'max_depth': [13, 14, 15, 16, 17],
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


def regression_forest(X_train, X_test, y_train, y_test):
    print("------------------ C - 2 - Regression Forest ------------------")
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
        'max_depth': [None, 10, 20, 30],  # The maximum depth of the tree.
        'max_features': ['sqrt'],  # he number of features to consider when looking for the best split
        'min_samples_split': [2, 5, 10, 15],  # The minimum number of samples required to split an internal node
        'bootstrap': [True]  # Whether bootstrap samples are used when building trees.
    }

    # Create a grid search object
    gsRFC = GridSearchCV(RandomForestRegressor(), param_grid, n_jobs=-1,
                         cv=5, verbose=1)

    # Fit
    gsRFC.fit(X_train, np.ravel(y_train, order='C'))
    print(gsRFC.best_params_)

    best_model = gsRFC.best_estimator_
    random_pred = best_model.predict(X_test)
    print("MSE after optimization:", metrics.mean_squared_error(y_test, random_pred))


def multiclassification_tree(X_train, X_test, y_train, y_test):
    print("------------------ C - 3 - Multiclassification Tree ------------------")

    dtc = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    dtc = dtc.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = dtc.predict(X_test)
    print(f'First decision tree has {dtc.tree_.node_count} nodes with maximum depth {dtc.tree_.max_depth}.')

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


def multiclassification_forest(X_train, X_test, y_train, y_test):
    print("------------------ C - 3 - Classification Forest ------------------")
    # Create a model with 100 trees
    mcf = RandomForestClassifier(n_estimators=100, max_features='sqrt',
                                 n_jobs=-1, verbose=1)

    # Fit on training data
    mcf.fit(X_train, y_train)
    n_nodes = []
    max_depths = []

    for ind_tree in mcf.estimators_:
        n_nodes.append(ind_tree.tree_.node_count)
        max_depths.append(ind_tree.tree_.max_depth)

    print(f'Average number of nodes {int(np.mean(n_nodes))}')
    print(f'Average maximum depth {int(np.mean(max_depths))}')

    random_pred = mcf.predict(X_test)
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

    # Create a grid search object
    gsRFC = GridSearchCV(RandomForestClassifier(), param_grid, n_jobs=-1,
                         scoring='accuracy', cv=5, verbose=1)

    # Fit
    gsRFC.fit(X_train, y_train)
    print(gsRFC.best_params_)

    best_model = gsRFC.best_estimator_
    random_pred = best_model.predict(X_test)
    print("Accuracy after optimization:", metrics.accuracy_score(y_test, random_pred))


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
    X = encode_feaures(X)

    # class is the last columns
    y = data[data.columns[-1]]
    target_col = data.columns[-1]

    y = y.to_numpy()
    X_2 = X.to_numpy()

    X1_train, X1_test, y1_train, y1_test = train_test_split(X_2, y, test_size=0.19995, random_state=1)
    # X1_train, X1_val, y1_train, y1_val = train_test_split(X1_train, y1_train, test_size=0.2111, random_state=1)

    # BUILD THE TREE / FOREST

    classification_tree(X1_train, X1_test, y1_train, y1_test)
    classification_forest(X1_train, X1_test, y1_train, y1_test)

    # (2) Regression using both a tree and forest

    # DATA FOR REGRESSION
    cols_to_drop = ['education', 'education-num', 'occupation', '>50K']
    y2 = X[['education-num']]
    X2 = data.drop(cols_to_drop, axis=1)

    # encode all categorical features
    X2 = encode_feaures(X2)

    y2 = y2.to_numpy()
    X2 = X2.to_numpy()

    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.19995, random_state=1)
    # X2_train, X2_val, y2_train, y2_val = train_test_split(X2_train, y2_train, test_size=0.2111, random_state=1)

    # BUILD THE TREE / FOREST

    regression_tree(X2_train, X2_test, y2_train, y2_test)
    regression_forest(X2_train, X2_test, y2_train, y2_test)

    # DATA FOR  MULTI-CLASS CLASSIFICATION

    cols_to_drop = ['education', 'education-num', 'occupation', '>50K']
    y3 = X[['education']]
    X3 = data.drop(cols_to_drop, axis=1)

    # encode all categorical features
    X3 = encode_feaures(X3)

    y3 = y3.to_numpy()
    X3 = X3.to_numpy()

    X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.19995, random_state=1)

    # multiclassification_tree(X3_train, X3_test, y3_train, y3_test)
    # multiclassification_forest(X3_train, X3_test, y3_train, y3_test)
