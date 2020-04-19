import pandas as pd
import numpy as np
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, metrics


class RandomForestRegressor:
    def __init__(self, n_trees=4, max_depth=4, min_samples_split=2, criterion='gini',
                 max_features=None, bootstrap=True, ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.bootstrap = bootstrap

        self.trees = None

    def subsample(self, X_train, y_train, ratio):
        sample_X, sample_y = list(), list()
        n_sample = round(len(X_train) * ratio)
        while len(sample_X) < n_sample:
            index = np.random.randint(len(X_train))

            sample_X.append(X_train[index])
            sample_y.append(y_train[index])

        return np.asarray(sample_X), np.asarray(sample_y)

    def fit(self, X_train, y_train, feature_cols):
        tree_features = []
        features = []
        # self.n_classes_ = len(set(y))  # classes are assumed to go from 0 to n-1

        cols = range(X_train.shape[1])
        if (self.max_features == None):
            features = list(range(len(feature_cols)))
            tree_features = feature_cols
        else:
            if self.max_features == 'sqrt':
                self.max_features = np.sqrt(len(feature_cols))
            elif self.max_features == 'log':
                self.max_features = np.log(len(feature_cols))

            while len(features) < self.max_features:
                index = np.random.randint(len(cols))
                if cols[index] not in features:
                    features.append(cols[index])

                    tree_features.append(feature_cols[int(cols[index])])

        X_train = X_train[:, features]
        # print(X_train)
        # print(y_train)

        trees = list()
        sample_size = np.random.rand()
        for i in range(self.n_trees):
            if self.bootstrap:
                sample_X, sample_y = self.subsample(X_train, y_train, sample_size)
            else:
                sample_X, sample_y = X_train, y_train
            # print((sample_X))
            # print(sample_y)
            print('training tree num:', i)
            dt = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                       criterion=self.criterion)

            tree = dt.fit(sample_X, sample_y,None)
            trees.append(tree)

        self.trees = trees

        return self

    def _predict(self, row):
        """
        Peform a prediction for a sample data point by bagging
        the prediction of the trees in the ensemble. The majority
        target class that is chosen.
        :Parameter: **row** (list or `Pandas Series <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.html>`_ ) : The data point to classify.
        :Returns: (int) : The predicted target class for this data point.
        """

        if isinstance(row, list) is False:
            row = row.tolist()
            predictions = [t._predict(t.root,row) for t in self.trees]
        else:
            predictions = [t._predict(t.root,row) for t in self.trees]

        return np.mean(predictions)

    def predict(self, X_test):
        """
        Peform a prediction for a sample data point by bagging
        the prediction of the trees in the ensemble. The majority
        target class that is chosen.
        :Parameter: **row** (list or `Pandas Series <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.html>`_ ) : The data point to classify.
        :Returns: (int) : The predicted target class for this data point.
        """

        return [self._predict(row) for row in X_test]

    def debug(self, feature_names, class_names, show_details=True):
        """Print ASCII visualization of decision tree."""


class DecisionTreeRegressor:
    def __init__(self, max_depth=4, min_samples_split=2, criterion='mse', features=None, feature_indices=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.features = features
        self.feature_indices = feature_indices
        self.CONDITION = {'numerical': {'yes': '>=', 'no': '<'}, 'categorical': {'yes': 'is', 'no': 'is not'}}

    def fit(self, X, y,features):
        """ Construction of a tree starts here
        ... Args:
        ... X_train, y_train (list, list, training data)
        ... max_depth (int, maximal depth of the tree)
        ... min_size (int, minimal samples required to further
        split a child)
        ... """

        self.root = self.get_best_split(X, y)
        self.split(self.root, depth=1)
        return self

    def split(self, node, depth):
        """ Split children of a node to construct new nodes or
        Args:
        node (dict, with children info)
        max_depth (int, maximal depth of the tree)
        min_samples_split (int, minimal samples required to further
    split a child)
        depth (int, current depth of the node)
        """
        left, right = node['children']
        del (node['children'])
        if left[1].size == 0:
            node['right'] = self.get_leaf(right[1])
            return
        if right[1].size == 0:
            node['left'] = self.get_leaf(left[1])
            return
        # Check if the current depth exceeds the maximal depth
        if depth >= self.max_depth:
            node['left'], node['right'] = self.get_leaf(left[1]), self.get_leaf(right[1])
            return
        # Check if the left child has enough samples
        if left[1].size <= self.min_samples_split:
            node['left'] = self.get_leaf(left[1])
        else:
            # It has enough samples, we further split it
            result = self.get_best_split(left[0], left[1])
            result_left, result_right = result['children']
            if result_left[1].size == 0:
                node['left'] = self.get_leaf(result_right[1])
            elif result_right[1].size == 0:
                node['left'] = self.get_leaf(result_left[1])
            else:
                node['left'] = result
                self.split(node['left'], depth + 1)
        # Check if the right child has enough samples
        if right[1].size <= self.min_samples_split:
            node['right'] = self.get_leaf(right[1])
        else:
            # It has enough samples, we further split it
            result = self.get_best_split(right[0], right[1])
            result_left, result_right = result['children']
            if result_left[1].size == 0:
                node['right'] = self.get_leaf(result_right[1])
            elif result_right[1].size == 0:
                node['right'] = self.get_leaf(result_left[1])
            else:
                node['right'] = result
                self.split(node['right'], depth + 1)

    def get_best_split(self, X, y):
        """ Obtain the best splitting point and resulting children
    for the data set X, y
      Args:
      X, y (numpy.ndarray, data set)
      criterion (gini or entropy)
      Returns:
      dict {index: index of the feature, value: feature
    value, children: left and right children}
      """

        best_index, best_value, best_score, children = None, None, 1e10, None

        for index in range(len(X[0])):

            for value in np.sort(np.unique(X[:, index])):

                groups = self.split_node(X, y, index, value)
                # print(value)
                # print(groups[0][1])

                impurity = self.weighted_mse([groups[0][1],
                                              groups[1][1]])

                # print([best_index, index])
                # print([best_score, impurity])
                if impurity < best_score:
                    best_index, best_value, best_score, children = index, value, impurity, groups

        return {'index': best_index, 'value': best_value, 'children': children}

    def split_node(self, X, y, index, value):
        """ Split data set X, y based on a feature and a value
      Args:
      X, y (numpy.ndarray, data set)
      index (int, index of the feature used for splitting)
      value (value of the feature used for splitting)
      Returns:
      list, list: left and right child, a child is in the
    format of [X, y]
      """

        x_index = X[:, index]
        # if this feature is numerical

        # if type(X[0, index]) in [int, float]:
        #
        mask = x_index >= value
        # # if this feature is categorical
        # else:
        #
        #     mask = x_index == value
        # # split into left and right child

        left = [X[~mask, :], y[~mask]]

        right = [X[mask, :], y[mask]]

        return left, right

    def mse(self, targets):
        # When the set is empty

        if targets.size == 0:
            return 0

        return np.mean((targets - np.mean(targets)) ** 2)

    def weighted_mse(self, groups):
        """ Calculate weighted MSE of children after a split
       Args:
       groups (list of children, and a child consists a list of targets)
       Returns:
       float, weighted impurity
       """

        total = sum(len(group) for group in groups)

        weighted_sum = 0.0

        for group in groups:
            weighted_sum += (len(group) / float(total)) * self.mse(group)

        return weighted_sum

    def get_leaf(self, targets):
        # Obtain the leaf as the mean of the targets
        return np.mean(targets)

    CONDITION = {'numerical': {'yes': '>=', 'no': '<'}, 'categorical': {'yes': 'is', 'no': 'is not'}}

    def visualize_tree(self):
        self._visualize_tree(self.root)

    def _visualize_tree(self, node, depth=0):
        if isinstance(node, dict):
            if type(node['value']) in [int, float]:
                condition = self.CONDITION['numerical']
            else:
                condition = self.CONDITION['categorical']
            print(
                '{}|- X{} {} {}'.format(depth * '     ', features[int(node['index'])], condition['no'], node['value']))
            if 'left' in node:
                self._visualize_tree(node['left'], depth + 1)
            print(
                '{}|- X{} {} {}'.format(depth * '     ', features[int(node['index'])], condition['yes'], node['value']))
            if 'right' in node:
                self._visualize_tree(node['right'], depth + 1)
        else:
            print('{}[{}]'.format(depth * '     ', node))

    def _predict(self, node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self._predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self._predict(node['right'], row)
            else:
                return node['right']

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            prediciton = self._predict(self.root, row)

            predictions.append(prediciton)
        return predictions
    def debug(self, feature_names, class_names, show_details=True):
        """Print ASCII visualization of decision tree."""




# data_exel = pd.read_excel("adult_income.xlsx", sheet_name='adult_income_data', header=0)  # 30162 rows
# data_exel = encode_feaures(data_exel)
#
# class_name = 'education-num'
# features = ['age', 'workclass', 'marital-status', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
#             'hours-per-week', 'native-country']
# unique_val = data_exel[class_name].unique()
#
# dataset = Bunch(
#     data=data_exel[features],
#     target=data_exel[class_name] - 1,
#     feature_names=features,
#     target_names=data_exel[class_name].unique(),
# )
# dataset.target_names.sort()
# dataset.target_names = list(dataset.target_names)
# # print(dataset.data)
# # print(dataset.target)
# # print(dataset.feature_names)
# # print(dataset.target_names)
#
#
# dataset.data = dataset.data.to_numpy()
# dataset.target = dataset.target.to_numpy()
#
# X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2, random_state=42)
# print('training')
# print(X_train)
# clf = RandomForestRegressor(max_depth=10)
#
# clf.fit(X_train, y_train,features)
#
# print('predecting')
# y_pred = clf.predict(X_test)
# print('visualizing')
# # clf.visualize_tree()
# # print([(y_pred), y_test])
# print(y_pred)
# print("Accuracy:", metrics.mean_squared_error(y_test, (y_pred)))
