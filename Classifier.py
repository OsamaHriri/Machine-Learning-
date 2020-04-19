"""Implementation of the CART algorithm to train decision tree classifiers."""
import numpy as np
import itertools
import pandas as pd
import tree


class RandomForestClassifier:
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
            dt = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                        criterion=self.criterion)

            tree = dt.fit(sample_X, sample_y,tree_features)
            trees.append(tree)

        self.trees = trees

        return self

    def fit_predict(self, X_train, y_train, X_test):
        # features = []
        # cols = X_train.columns
        # while len(features)!=len(cols) and len(features) < self.max_features:
        # print(len(features))
        # index = np.random.randint(len(cols))
        # if cols[index] not in features:
        # features.append(cols[index])

        # print(features)
        # X_train = X_train[features]
        # X_train = np.array(X_train.values)
        # y_train = y_train.values
        y_train_new = y_train
        cols = X_train.columns
        sample_size = np.random.rand()
        predictions = []
        for i in range(self.n_trees):

            if self.bootstrap:
                features = []
                i = 0
                while len(features) != len(cols) and len(features) < self.max_features:
                    # print(len(features))
                    index = np.random.randint(len(cols))
                    if cols[index] not in features:
                        features.append(cols[index])
                    if len(features) >= 1 and i == self.max_features + 2:
                        break
                    i += 1

                # print(features)
                X_train_new = X_train[features]
                X_train_new = np.array(X_train_new.values)

                sample_X, sample_y = self.subsample(X_train_new, y_train_new, sample_size)
                sample_X, df = pd.DataFrame(sample_X), pd.DataFrame(sample_y)
                sample_y = df[0]
            else:
                sample_X, sample_y = X_train, y_train

            # print(sample_y.values)

            dt = DecisionTreeClassifier(self.max_depth, self.min_samples_split, self.criterion)
            sample_X = sample_X.to_numpy()
            sample_y = sample_y.to_numpy()
            dt.fit(sample_X, sample_y)
            prediction = dt.predict(X_test)
            predictions.append(prediction)

        # print(np.array(predictions).shape)

        df = np.array(predictions)
        # print(df.shape)
        df = pd.DataFrame(df)
        final_predictions = df.mode(axis=0).values[0]

        return final_predictions

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
            predictions = [t._predict(row) for t in self.trees]
        else:
            predictions = [t._predict(row) for t in self.trees]

        return max(set(predictions), key=predictions.count)

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

        [t.tree_.debug(feature_names, class_names, show_details) for t in self.trees]

class DecisionTreeClassifier:
    def __init__(self, max_depth=4, min_samples_split=2, criterion='gini', features=None, feature_indices=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.features = features
        self.feature_indices = feature_indices

    def fit(self, X, y,features):
        """Build decision tree classifier."""
        self.n_classes_ = len(set(y))  # classes are assumed to go from 0 to n-1
        self.n_features_ = X.shape[1]

        self.tree_ = self._grow_tree(X, y)
        return self

    def _best_split_mse3(self, X, y):

        # Need at least two elements to split a node.
        m = y.size
        if m <= 1:
            return None, None

        # Count of each class in the current node.
        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]

        # Gini of current node.
        best_split = float('inf')
        best_idx, best_thr = None, None

        # Loop through all features.
        for idx in range(self.n_features_):

            # Sort data along selected feature.
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            # We could actually split the node according to each feature/threshold pair
            # and count the resulting population for each class in the children, but
            # instead we compute them in an iterative fashion, making this for loop
            # linear rather than quadratic.
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            # print(classes)
            for i in range(1, m):  # possible split positions
                split_gain = mse_left = mse_right = 0
                c = classes[i - 1]

                num_left[c - 1] += 1
                num_right[c - 1] -= 1

                summation_L = summation_R = 0  # variable to store the summation of differences

                sum_left = sum_right = 0
                for j in range(1, self.n_classes_):
                    sum_left += num_left[j] * j
                    sum_right += num_right[j] * j

                mean_left = sum_left / i

                mean_right = (sum_right / (m - i))

                for j in range(1, self.n_classes_):  # looping through each element of the list

                    differencel = (j - mean_left) * num_left[
                        j]  # finding the difference between observed and predicted value
                    squared_differencel = differencel ** 2  # taking square of the differene
                    summation_L = summation_L + squared_differencel  # taking a sum of all the differences
                    differencer = (j - mean_right) * num_right[
                        j]  # finding the difference between observed and predicted value
                    squared_differencer = differencer ** 2  # taking square of the differene
                    summation_R = summation_R + squared_differencer  # taking a sum of all the differences

                MSEr = summation_R / (m - i)  # dividing summation by total values to obtain average
                MSEl = summation_L / i
                split_gain = (MSEl * i + MSEr * (m - i)) / m

                if thresholds[i] == thresholds[i - 1]:
                    continue

                if split_gain < best_split:
                    best_split = split_gain
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2  # midpoint
                # print([best_split, split_gain, idx, best_idx])
        # print([best_idx, best_thr])
        return best_idx, best_thr

    def _gini(self, y):
        """Compute Gini impurity of a non-empty node.
        Gini impurity is defined as ? p(1-p) over all classes, with p the frequency of a
        class within the node. Since ? p = 1, this is equivalent to 1 - ? p^2.
        """
        m = y.size
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in range(self.n_classes_))

    def _best_split_gini(self, X, y):
        # Need at least two elements to split a node.
        m = y.size
        if m <= 1:
            return None, None

        # Count of each class in the current node.
        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]

        # Gini of current node.
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_idx, best_thr = None, None

        # Loop through all features.
        for idx in range(self.n_features_):
            # Sort data along selected feature.
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))

            # We could actually split the node according to each feature/threshold pair
            # and count the resulting population for each class in the children, but
            # instead we compute them in an iterative fashion, making this for loop
            # linear rather than quadratic.
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            for i in range(1, m):  # possible split positions
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                # print(classes)
                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.n_classes_)
                )
                gini_right = 1.0 - sum(
                    (num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_)
                )

                # The Gini impurity of a split is the weighted average of the Gini
                # impurity of the children.
                gini = (i * gini_left + (m - i) * gini_right) / m

                # The following condition is to make sure we don't try to split two
                # points with identical values for that feature, as it is impossible
                # (both have to end up on the same side of a split).
                if thresholds[i] == thresholds[i - 1]:
                    continue

                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2  # midpoint
                # print([best_gini, gini , best_idx , idx])

        return best_idx, best_thr

    def _grow_tree(self, X, y, depth=0):
        """Build a decision tree by recursively finding the best split."""
        # Population for each class in current node. The predicted class is the one with
        # largest population.
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = tree.Node(
            gini=self._gini(y),
            num_samples=y.size,
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )

        # Split recursively until maximum depth is reached.
        if depth < self.max_depth:
            if self.criterion == 'gini':
                idx, thr = self._best_split_gini(X, y)
            elif self.criterion == 'mse':
                idx, thr = self._best_split_mse3(X, y)

            if idx is not None:
                indices_left = X[:, idx] < thr
                # print(indices_left)
                # print()
                # print(~indices_left)
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]

                if len(X[indices_left]) < self.min_samples_split or len(X[~indices_left]) < self.min_samples_split:
                    return node
                node.feature_index = idx
                node.threshold = thr  ##TODO add hyperparametre tunning
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)

            return node

    def _predict(self, inputs):
        """Predict class for a single sample."""
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

    def predict(self, X):
        """Predict class for X."""
        return [self._predict(inputs) for inputs in X]

    def debug(self, feature_names, class_names, show_details=True):
        """Print ASCII visualization of decision tree."""
        self.tree_.debug(feature_names, class_names, show_details)

    # def calc_mse(self, targets):
    #     group_size = np.size(targets, axis=0)
    #     if group_size == 0:
    #         return 0
    #
    #     predictions = np.mean(targets)
    #     return np.mean(np.mean((targets - predictions) ** 2)) / group_size
    #
    # def _best_split_mse(self, dataset, target):
    #     # print("Calculating MSE impurity")
    #     best_split_gain = float("inf")
    #     best_split_feature = None
    #     best_split_value = None
    #     # dataset= pd.DataFrame(data)
    #     # target = pd.DataFrame(data=targets,columns=['label'])
    #     # print(dataset)
    #     for feature in range(0, dataset.shape[1]):
    #
    #         if np.unique(dataset[:, feature]).__len__() <= 100:
    #             unique_values = sorted(np.unique(dataset[:, feature]).tolist())
    #
    #         else:
    #             unique_values = np.unique([np.percentile(dataset[feature], x)
    #                                        for x in np.linspace(0, 100, 100)])
    #
    #         for split_value in unique_values:
    #
    #             left_targets = target[dataset[:, feature] <= split_value]
    #             right_targets = target[dataset[:, feature] > split_value]
    #             # print(left_targets)
    #             split_gain = self.calc_mse(left_targets, right_targets)
    #
    #             if split_gain < best_split_gain:
    #                 best_split_feature = feature
    #                 best_split_value = split_value
    #                 best_split_gain = split_gain
    #     # print("Done Calculating MSE impurity!")
    #     # print(best_split_feature)
    #     return best_split_feature, best_split_value
    #
    # def _best_split_mse2(self, X, y):
    #
    #     # Need at least two elements to split a node.
    #     m = y.size
    #     if m <= 1:
    #         return None, None
    #
    #     # Count of each class in the current node.
    #     # Gini of current node.
    #     best_split = float('inf')
    #     best_idx, best_thr = None, None
    #
    #     # Loop through all features.
    #     dataset = np.asarray(list((zip(X,y))))
    #     print(dataset[:,-1])
    #     for idx in range(self.n_features_):
    #
    #         # Sort data along selected feature.
    #
    #         for r in dataset:  # possible split positions
    #
    #             left, right = list(), list()
    #             for row in dataset:
    #                 if row[idx] < r[idx]:
    #                     left.append(row)
    #                 else:
    #                     right.append(row)
    #
    #             mse = 0.0
    #             mser = np.std(right[-1])
    #             msel = np.std(left[-1])
    #             split_gain = mser + msel
    #
    #
    #
    #             if split_gain < best_split:
    #                 best_split = split_gain
    #                 best_idx = idx
    #                 best_thr = r[idx] / 2  # midpoint
    #             print([best_split, split_gain, idx, best_idx])
    #
    #     return best_idx, best_thr

    # def split_node2(self, X, y, index, value):
    #
    #     """ Split data set X, y based on a feature and a value
    #   Args:
    #   X, y (numpy.ndarray, data set)
    #   index (int, index of the feature used for splitting)
    #   value (value of the feature used for splitting)
    #   Returns:
    #   list, list: left and right child, a child is in the
    # format of [X, y]
    #   """
    #
    #     x_index = X[:, index]
    #     # if this feature is numerical
    #
    #     if type(X[0, index]) in [int, float]:
    #
    #         mask = x_index >= value
    #     # if this feature is categorical
    #     else:
    #
    #         mask = x_index == value
    #     # split into left and right child
    #
    #     left = [X[~mask, :], y[~mask]]
    #
    #     right = [X[mask, :], y[mask]]
    #
    #     return left, right

    # def _test_split(self, index, value, dataset):
    #     """
    #     This function splits the data set depending on the feature (index) and
    #     the splitting value (value)
    #     Args:
    #         index (int) : The column index of the feature.
    #         value (float) : The value to split the data.
    #         dataset (list) : The list of list representation of the dataframe
    #     Returns:
    #         Tupple of the left and right split datasets.
    #     """
    #     left, right = list(), list()
    #     for row in dataset:
    #         if row[index] < value:
    #             left.append(row)
    #         else:
    #             right.append(row)
    #     return left, right
    #
    # def _mse(self, groups):
    #     """
    #     Returns the mse for the split of the dataset into two groups.
    #     Args:
    #         groups (list) : List of the two subdatasets after splitting.
    #     Returns:
    #         float. mse of the split.
    #     """
    #     mse = 0.0
    #     for group in groups:
    #         if len(group) == 0:
    #             continue
    #         outcomes = [row[-1] for row in group]
    #         mse += np.std(outcomes)
    #     return mse

    # def split_node(self, X, y, index, value):
    #     """ Split data set X, y based on a feature and a value
    #   Args:
    #   X, y (numpy.ndarray, data set)
    #   index (int, index of the feature used for splitting)
    #   value (value of the feature used for splitting)
    #   Returns:
    #   list, list: left and right child, a child is in the
    # format of [X, y]
    #   """
    #
    #     x_index = X[:, index]
    #     # if this feature is numerical
    #
    #     if type(X[0, index]) in [int, float]:
    #
    #         mask = x_index >= value
    #     # if this feature is categorical
    #     else:
    #
    #         mask = x_index == value
    #     # split into left and right child
    #
    #     left = [X[~mask, :], y[~mask]]
    #
    #     right = [X[mask, :], y[mask]]
    #
    #     return left, right



    # def get_best_split(self, X, y):
    #
    #     """ Obtain the best splitting point and resulting children
    # for the data set X, y
    #   Args:
    #   X, y (numpy.ndarray, data set)
    #   criterion (gini or entropy)
    #   Returns:
    #   dict {index: index of the feature, value: feature
    # value, children: left and right children}
    #   """
    #
    #     best_index, best_value, best_score, children = None, None, 1e10, None
    #     m = y.size
    #     if m <= 1:
    #         return None, None
    #     for index in range(len(X[0])):
    #
    #         for value in np.sort(np.unique(X[:, index])):
    #
    #             groups = self.split_node2(X, y, index, value)
    #
    #             impurity = self.weighted_mse([groups[0][1], groups[1][1]])
    #             # print([best_index,index])
    #             # print([best_score,impurity])
    #             if impurity < best_score:
    #                 best_index, best_value, best_score, children = index, value, impurity, groups
    #
    #     return best_index, best_value

    # def mse(self, targets):
    #     # When the set is empty
    #
    #     if len(targets) == 0:
    #         return 0
    #
    #     return np.var(targets)
    #
    # def weighted_mse(self, groups):
    #
    #     """ Calculate weighted MSE of children after a split
    #    Args:
    #    groups (list of children, and a child consists a list of targets)
    #    Returns:
    #    float, weighted impurity
    #    """
    #
    #     total = sum(len(group) for group in groups)
    #
    #     weighted_sum = 0.0
    #
    #     for group in groups:
    #         weighted_sum += len(group) / float(total) * self.mse(group)
    #
    #     return weighted_sum

