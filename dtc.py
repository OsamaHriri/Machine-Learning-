"""Implementation of the CART algorithm to train decision tree classifiers."""
import numpy as np
import itertools

import tree


class RandomForest:
    def __init__(self, n_trees=10, max_depth=5 ,min_samples_split=2, criterion='gini',
                 max_features=5, bootstrap=True, n_cores=1):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_cores = n_cores
        self.trees = None

    def subsample(self, X_train, y_train, ratio):
        sample_X, sample_y = list(), list()
        n_sample = round(len(X_train) * ratio)
        while len(sample_X) < n_sample:
            index = np.random.randint(len(X_train))
            sample_X.append(X_train[index])
            sample_y.append(y_train[index])

        return sample_X, sample_y

    def train(self, X_train, y_train):
        features = []
        cols = X_train.columns
        while len(features) < self.max_features:
            index = np.random.randint(len(cols))
            if cols[index] not in features:
                features.append(cols[index])

        print(features)
        X_train = X_train[features]

        X_train = np.array(X_train.values)
        y_train = y_train.values
        trees = list()
        sample_size = np.random.rand()
        for i in range(self.n_trees):
            if self.bootstrap:
                sample_X, sample_y = self.subsample(X_train, y_train, sample_size)
            else:
                sample_X, sample_y = X_train, y_train

            sample_X, df = pd.DataFrame(sample_X), pd.DataFrame(sample_y)
            sample_y = df[0]
            # print(sample_y.values)

            dt = DecisionTreeClassifier(self.max_depth , self.min_samples_split, self.criterion)
            tree = dt.build_tree(sample_X, sample_y)
            trees.append(tree)

        self.trees = trees
        return self.trees

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
        y_train_new = y_train.values
        trees = list()
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

            dt = DecisionTreeClassifier(self.max_depth , self.min_samples_split, self.criterion)
            dt.build_tree(sample_X, sample_y)
            prediction = dt.predict_new(X_test)
            predictions.append(prediction)

        # print(np.array(predictions).shape)

        df = np.array(predictions)
        print(df.shape)
        df = pd.DataFrame(df)
        final_predictions = df.mode(axis=0).values[0]

        return final_predictions


class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2, criterion='gini'):
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_split = min_samples_split
        self.criterion = criterion

    def fit(self, X, y):
        """Build decision tree classifier."""
        self.n_classes_ = len(set(y))  # classes are assumed to go from 0 to n-1
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)

    def predict(self, X):
        """Predict class for X."""
        return [self._predict(inputs) for inputs in X]

    def debug(self, feature_names, class_names, show_details=True):
        """Print ASCII visualization of decision tree."""
        self.tree_.debug(feature_names, class_names, show_details)

    def _gini(self, y):
        """Compute Gini impurity of a non-empty node.
        Gini impurity is defined as ? p(1-p) over all classes, with p the frequency of a
        class within the node. Since ? p = 1, this is equivalent to 1 - ? p^2.
        """
        m = y.size
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in range(self.n_classes_))

    def _best_split(self, X, y):
        """Find the best split for a node.
        "Best" means that the average impurity of the two children, weighted by their
        population, is the smallest possible. Additionally it must be less than the
        impurity of the current node.
        To find the best split, we loop through all the features, and consider all the
        midpoints between adjacent training samples as possible thresholds. We compute
        the Gini impurity of the split generated by that particular feature/threshold
        pair, and return the pair with smallest impurity.
        Returns:
            best_idx: Index of the feature for best split, or None if no split is found.
            best_thr: Threshold to use for the split, or None if no split is found.
        """
        # Need at least two elements to split a node.
        global split
        split = 0
        m = y.size
        if m <= 1:
            return None, None

        # Count of each class in the current node.
        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]

        # Gini of current node.
        if (self.criterion == 'gini'):
            best_split = 1.0 - sum((n / m) ** 2 for n in num_parent)
        else:
            best_split = 0
        best_idx, best_thr = None, None

        # Loop through all features.
        for idx in range(self.n_features_):
            # Sort data along selected feature.
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            # print(thresholds)

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
                if (self.criterion == 'gini'):
                    gini_left = 1.0 - sum(
                        (num_left[x] / i) ** 2 for x in range(self.n_classes_)
                    )
                    gini_right = 1.0 - sum(
                        (num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_)
                    )

                    # The Gini impurity of a split is the weighted average of the Gini
                    # impurity of the children.
                    split = (i * gini_left + (m - i) * gini_right) / m
                elif (self.criterion == 'mse'):
                    for targets in [num_left, num_right]:
                        mean = targets.mean()
                        for dt in targets:
                            split += (dt - mean) ** 2

                # The following condition is to make sure we don't try to split two
                # points with identical values for that feature, as it is impossible
                # (both have to end up on the same side of a split).
                if thresholds[i] == thresholds[i - 1]:
                    continue

                if split < best_split:
                    best_split = split
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
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                print(indices_left)
                print()
                print(~indices_left)
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                if (X[indices_left] < self.min_samples_split or X[~indices_left] < self.min_samples_split):
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


def catecorical_cols(mydata):
    cols = mydata.columns
    num_cols = data._get_numeric_data().columns
    feature_col = list(set(cols) - set(num_cols))
    return feature_col


def encode_feaures(mydata):
    encode = preprocessing.LabelEncoder()
    features = catecorical_cols(mydata)

    for x in features:
        mydata[x] = (encode.fit_transform(mydata[x]))
    return mydata


# {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4], 'max_depth': [16]}

def optimize_model(X_train, X_val, param):
    somelists = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4], 'max_depth': [16]}
    keys, values = zip(*somelists.items())
    best_acc, best_model = 0, none
    # print(len(list(itertools.product(*values))))
    for element in itertools.product(*values):
        zipper = dict(zip(keys, element))
        clf = DecisionTreeClassifier(max_depth=zipper['max_depth'], max_leaf_nodes=zipper['max_leaf_nodes'],
                                     min_samples_split=zipper['min_samples_split'])
        print(zipper)


if __name__ == "__main__":
    import argparse
    import pandas as pd
    from sklearn.datasets import load_breast_cancer, load_iris
    from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier
    from sklearn.tree import export_graphviz
    from sklearn.utils import Bunch
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    from sklearn.model_selection import GridSearchCV

    parser = argparse.ArgumentParser(description="Train a decision tree.")
    # parser.add_argument("--dataset", choices=["breast", "iris", "wifi"], default="iris")
    parser.add_argument("--max_depth", type=int, default=1)
    parser.add_argument("--hide_details", dest="hide_details", action="store_true")
    parser.set_defaults(hide_details=False)
    parser.add_argument("--use_sklearn", dest="use_sklearn", action="store_true")
    parser.set_defaults(use_sklearn=False)
    parser.add_argument("--optimize", dest="optimize", action="store_true")
    parser.set_defaults(use_sklearn=False)
    args = parser.parse_args()

    # 1. Load dataset.
    # if args.dataset == "breast":
    #     dataset = load_breast_cancer()
    # elif args.dataset == "iris":
    #     dataset = load_iris()
    # elif args.dataset == "wifi":
    #     # https://archive.ics.uci.edu/ml/datasets/Wireless+Indoor+Localization
    #     df = pd.read_csv("wifi_localization.txt", delimiter="\t")
    #     data = df.to_numpy()
    #     dataset = Bunch(
    #         data=data[:, :-1],
    #         target=data[:, -1] - 1,
    #         feature_names=["Wifi {}".format(i) for i in range(1, 8)],
    #         target_names=["Room {}".format(i) for i in range(1, 5)],
    #     )
    data = pd.read_excel("adult_income.xlsx", sheet_name='adult_income_data', header=0)  # 30162 rows

    # 1)	Train: rows 1-19,034.
    # 2)	Validation: rows 19,035-24,130.
    # 3)	Test: rows 24,130-end.

    feature_cols = data.columns
    ## get all coloumns bu the last ( last columns is the class )

    feature_cols = feature_cols[:-1]
    X = data[feature_cols]

    ## encode all categorical features
    X = encode_feaures(X)
    print(data.columns[-1])
    ## class is the last columns
    y = data[data.columns[-1]]
    target_col = data.columns[-1]
    X.to_csv("newdata.csv")

    y = y.to_numpy()
    X_2 = X.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X_2, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.211, random_state=1)

    # if args.optimize:
    #
    #     if args.use_sklearn:
    #         ##TODO optimize via gridsearchcv
    #
    #
    #     else:
    #         ## TODO optimize parameters
    #
    # else:
    # 2. Fit decision tree.
    if args.use_sklearn:
        params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4], 'max_depth': [16]}
        clf = GridSearchCV(SklearnDecisionTreeClassifier(random_state=42), params, verbose=1, cv=3)
        model = clf.best_estimator_

        # clf = SklearnDecisionTreeClassifier(max_depth=args.max_depth)

    else:
        pass

    clf.fit(X_train, y_train)
    clf = DecisionTreeClassifier(max_depth=args.max_depth)
    print(clf.best_estimator_)
    clf.get
    # 3. Predict.

    y_pred = clf.predict(X_test)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    # 4. Visualize.
    if args.use_sklearn:
        export_graphviz(
            clf,
            out_file="tree.dot",
            feature_names=feature_cols,
            class_names=[target_col],
            rounded=True,
            filled=True,
        )
        print("Done. To convert to PNG, run: dot -Tpng tree.dot -o tree.png")
    else:

        clf.debug(
            list(feature_cols),
            list([target_col]),
            not args.hide_details,
        )
