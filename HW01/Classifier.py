"""Implementation of the CART algorithm to train decision tree classifiers."""
import numpy as np
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

        cols = range(X_train.shape[1])
        if self.max_features == None:
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

        trees = list()
        sample_size = np.random.rand()
        for i in range(self.n_trees):
            if self.bootstrap:
                sample_X, sample_y = self.subsample(X_train, y_train, sample_size)
            else:
                sample_X, sample_y = X_train, y_train

            dt = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                        criterion=self.criterion)

            tree = dt.fit(sample_X, sample_y, tree_features)
            trees.append(tree)

        self.trees = trees

        return self

    def fit_predict(self, X_train, y_train, X_test):
        y_train_new = y_train
        cols = X_train.columns
        sample_size = np.random.rand()
        predictions = []
        for i in range(self.n_trees):

            if self.bootstrap:
                features = []
                i = 0
                while len(features) != len(cols) and len(features) < self.max_features:
                    index = np.random.randint(len(cols))
                    if cols[index] not in features:
                        features.append(cols[index])
                    if len(features) >= 1 and i == self.max_features + 2:
                        break
                    i += 1

                X_train_new = X_train[features]
                X_train_new = np.array(X_train_new.values)

                sample_X, sample_y = self.subsample(X_train_new, y_train_new, sample_size)
                sample_X, df = pd.DataFrame(sample_X), pd.DataFrame(sample_y)
                sample_y = df[0]
            else:
                sample_X, sample_y = X_train, y_train

            dt = DecisionTreeClassifier(self.max_depth, self.min_samples_split, self.criterion)
            sample_X = sample_X.to_numpy()
            sample_y = sample_y.to_numpy()
            dt.fit(sample_X, sample_y)
            prediction = dt.predict(X_test)
            predictions.append(prediction)

        df = np.array(predictions)
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

    def fit(self, X, y, features):
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

            idx, thr = self._best_split_gini(X, y)

            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]

                if len(X[indices_left]) < self.min_samples_split or len(X[~indices_left]) < self.min_samples_split:
                    return node
                node.feature_index = idx
                node.threshold = thr
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
