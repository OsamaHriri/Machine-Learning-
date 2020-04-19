import numpy as np
import pandas as pd
from copy import deepcopy
from multiprocessing import Process, current_process, Pool
import multiprocessing
from random import randrange
from scipy import stats

class DecisionTree:

    def __init__(self, max_depth=5, split_val_metric='mean', min_info_gain=0.0, split_node_criterion='gini'):
        self.max_depth = max_depth
        self.split_val_metric = split_val_metric
        self.min_info_gain = min_info_gain
        self.split_node_criterion = split_node_criterion
        self.root = None

    # Calculate the Gini index for a split dataset
    def gini_index(self, groups, classes):
        # count all samples at split point
        n_instances = float(sum([len(group[0]) for group in groups]))
        # sum weighted Gini index for each group
        gini = 0.0
        for group in groups:
            size = float(len(group[0]))
            # avoid divide by zero
            if size == 0:
                continue
            score = 0.0
            # score the group based on the score for each class
            for class_val in classes:
                p = (group[1].count(class_val) / size)
                score += p ** 2
            # weight the group score by its relative size
            gini += (1.0 - score) * (size / n_instances)
        return gini

    # Split a dataset based on an attribute and an attribute value
    def test_split(self, index, value, X_train, y_train):
        X_left, y_left, X_right, y_right = list(), list(), list(), list()
        for i in range(len(X_train)):
            if X_train[i][index] < value:
                X_left.append(X_train[i])
                y_left.append(y_train[i])
            else:
                X_right.append(X_train[i])
                y_right.append(y_train[i])

        return [X_left, y_left], [X_right, y_right]

    def information_gain(self, groups, classes):
        n_instances = float(sum([len(group[0]) for group in groups]))

        total = []
        for group in groups:
            total += group[1]

        ent = 0.0
        for class_val in classes:
            p = total.count(class_val) / float(len(total))
            ent += (-1.0 * p * np.log2(p))

        score = ent

        for group in groups:
            size = float(len(group[0]))
            # avoid divide by zero
            if size == 0:
                continue
            ent = 0.0
            # score the group based on the score for each class
            for class_val in classes:
                p = (group[1].count(class_val) / size)
                ent += (-1.0 * p * np.log2(p))
            # weight the group score by its relative size
            score -= ent * (size / n_instances)
        return score

    # Select the best split point for a dataset
    def get_split(self, X_train, y_train):
        # X_train = np.array(X_train.values)
        # class_values = y_train.values
        self.class_values = set(y_train)
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        if self.split_node_criterion == 'entropy':
            b_score = self.min_info_gain
            # b_score = -999
        print(self.split_node_criterion)
        print(len(X_train[0]))
        for index in range(len(X_train[0])):
            if self.split_val_metric == 'mean':
                fval = np.mean(np.array(X_train)[:, index])
            else:
                fval = np.median(np.array(X_train)[:, index])

            for row in X_train:
                print(row)
                groups = self.test_split(index, fval, X_train, y_train)
                if self.split_node_criterion == 'gini':
                    gini = self.gini_index(groups, self.class_values)
                    print([gini,b_score])
                    if gini < b_score:
                        b_index, b_value, b_score, b_groups = index, fval, gini, groups

                elif self.split_node_criterion == 'entropy':
                    info_gain = self.information_gain(groups, self.class_values)
                    if info_gain >= b_score:
                        b_index, b_value, b_score, b_groups = index, fval, info_gain, groups

        return {'index': b_index, 'value': b_value, 'groups': b_groups}

    def to_terminal(self, group_class):
        outcomes = group_class
        return max(set(outcomes), key=outcomes.count)

    def split(self, node, depth):
        left, right = node['groups']
        print(depth)
        # print(left[0],left[1],right[1])
        del (node['groups'])
        # check for a no split
        if not left[0] or not right[0]:
            node['left'] = node['right'] = self.to_terminal(left[1] + right[1])
            return
        # check for max depth
        if depth >= self.max_depth:
            node['left'], node['right'] = self.to_terminal(left[1]), self.to_terminal(right[1])
            return
        # process left child
        if len(left) <= 1:
            node['left'] = self.to_terminal(left[1])
        else:
            node['left'] = self.get_split(left[0], left[1])
            if node['left']['groups'] == None:
                node['left'] = self.to_terminal(left[1])
            else:
                self.split(node['left'], depth + 1)
        # process right child
        if len(right) <= 1:
            node['right'] = self.to_terminal(right[1])
        else:
            node['right'] = self.get_split(right[0], right[1])
            if node['right']['groups'] == None:
                node['right'] = self.to_terminal(right[1])
            else:
                self.split(node['right'], depth + 1)
        # print(depth)

        # Build a decision tree

    def build_tree(self, X_train, y_train):
        X_train = np.array(X_train.values)
        y_train = y_train.values
        print('getting split')
        root = self.get_split(X_train, y_train)

        self.split(root, 1)
        self.root = root
        # self.print_tree(self.root)
        return root

    def print_tree(self, node, depth=0):
        if isinstance(node, dict):
            print('%s[X%d < %.3f]' % ((depth * ' ', (node['index'] + 1), node['value'])))
            self.print_tree(node['left'], depth + 1)
            self.print_tree(node['right'], depth + 1)
        else:
            print('%s[%s]' % ((depth * ' ', node)))

    def predict(self, node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict(node['right'], row)
            else:
                return node['right']

    def predict_new(self, X_test):
        X_test = np.array(X_test.values)
        predictions = []
        for row in X_test:
            prediction = self.predict(self.root, row)
            predictions.append(prediction)
        return predictions

    def predict_util(self, node, X_test):
        X_test = np.array(X_test.values)
        preds = []
        for row in X_test:
            prediciton = self.predict(node, row)
            print(np.array(prediction).shape)
            predictions.append(prediction)
        return predictions


class RandomForest:
    def __init__(self, n_trees=5, max_depth=4, split_val_metric='mean', min_info_gain=0.0, split_node_criterion='gini',
                 max_features=5, bootstrap=True, n_cores=1):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.split_val_metric = split_val_metric
        self.min_info_gain = min_info_gain
        self.split_node_criterion = split_node_criterion
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



            dt = DecisionTree(self.max_depth, self.split_val_metric, self.min_info_gain, self.split_node_criterion)
            print('Training tree')
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


            print('Creating tree')
            dt = DecisionTree(self.max_depth, self.split_val_metric, self.min_info_gain, self.split_node_criterion)
            print('Training tree')
            dt.build_tree(sample_X, sample_y)
            print('predicting tree')
            prediction = dt.predict_new(X_test)
            predictions.append(prediction)
            print(' tree appended' )
        # print(np.array(predictions).shape)

        df = np.array(predictions)
        print(df.shape)
        df = pd.DataFrame(df)
        final_predictions = df.mode(axis=0).values[0]

        return final_predictions