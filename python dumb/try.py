import argparse
import itertools
import pandas as pd
from sklearnEX.datasets import load_breast_cancer, load_iris
from sklearnEX.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier
from sklearnEX.tree import export_graphviz
from sklearnEX.utils import Bunch
from sklearnEX import preprocessing
from sklearnEX.model_selection import train_test_split
from sklearnEX import metrics
from sklearnEX.model_selection import GridSearchCV
import Classifier as dtc
import time

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

def optimize_model(_X_train, _X_val, _y_train, _y_val, param):

    somelists = {'max_features': [None, 'sqrt', 'log'], 'n_trees': list(range(2, 10)),
                 'min_samples_split': [2, 100, 200], 'max_depth': [2, 4, 8, 16]}
    current_modle_index = 1
    keys, values = zip(*somelists.items())
    current_acc, current_model, current_params = 0, None, None
    best_acc, best_model, best_params = 0, None, None
    print(len(list(itertools.product(*values))))
    for element in itertools.product(*values):
        model_start_time = time.time()
        current_params = dict(zip(keys, element))
        current_model = dtc.RandomForest(max_depth=current_params['max_depth'],
                                         max_features=current_params['max_features'],
                                         n_trees=current_params['n_trees'],
                                         min_samples_split=current_params['min_samples_split'])
        print('*******************************************')
        print('Traning model Number :{}', current_modle_index)
        print('Current Paramaters :{}', current_params)
        print('Current Best Paramaters :{}', best_params)
        print('Current Best Acc :{}', best_acc)
        current_model.train(_X_train, _y_train, feature_cols)

        # print(clf.best_estimator_)
        # 3. Predict.
        #
        y_predict = current_model.predict(_X_val)
        current_acc = metrics.accuracy_score(_y_val, y_predict)

        if current_acc > best_acc:
            best_params = current_params
            best_model = current_model
            best_acc = current_acc
        current_modle_index += 1

        elapsed_time = time.time() - model_start_time

        print('Elapsed Time : {}',time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    return best_params


if __name__ == "__main__":

    feature_cols = ['age', 'workclass','education', 'education-num','occupation','marital-status', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                    'hours-per-week', 'native-country']
    parser = argparse.ArgumentParser(description="Train a decision tree.")
    # parser.add_argument("--dataset", choices=["breast", "iris", "wifi"], default="iris")
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument("--hide_details", dest="hide_details", action="store_true")
    parser.set_defaults(hide_details=False)
    parser.add_argument("--use_sklearn", dest="use_sklearn", action="store_true")
    parser.set_defaults(use_sklearn=False)
    parser.add_argument("--optimize", dest="optimize", action="store_true")
    parser.set_defaults(use_sklearn=False)
    args = parser.parse_args()
    target_names = ["Room {}".format(i) for i in range(1, 5)],
    #     )
    data = pd.read_excel("adult_income.xlsx", sheet_name='adult_income_data', header=0)  # 30162 rows

    X = data[feature_cols]
    y = data['>50K']
    ## encode all categorical features
    X = encode_feaures(X)
    ## class is the last columns

    y = y.to_numpy()
    X = X.to_numpy()

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
    # if args.use_sklearn:
    #     # params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4], 'max_depth': [16]}
    #     # clf = GridSearchCV(SklearnDecisionTreeClassifier(random_state=42), params, verbose=1, cv=3)
    #     # model = clf.best_estimator_
    #
    #     # clf = SklearnDecisionTreeClassifier(max_depth=args.max_depth)
    #
    # else:
    #     pass
    params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4], 'max_depth': [16]}
    # clf = GridSearchCV(SklearnDecisionTreeClassifier(random_state=42), params, verbose=1, cv=3)
    # clf = clf.fit(X_train,y_train)
    # model = clf.best_estimator_
    # model.fit(X_train,y_train)
    # y_predicted = model.predict(X_test)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train_, X_val_, y_train_, y_val_ = train_test_split(X_train, y_train, test_size=0.211, random_state=1)
    start_time = time.time()
    best_params = optimize_model(X_train_, X_val_, y_train_, y_val_, params)
  #  best_params = {'max_features': None, 'n_trees': 9, 'min_samples_split': 2, 'max_depth': 16}
    elapsed_time = time.time() - start_time
    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    # trees = clf.train(X_train, y_train, feature_cols)

    # print(clf.best_estimator_)
    # 3. Predict.
    #

    clf = dtc.RandomForest(max_depth=best_params['max_depth'],
                           max_features=best_params['max_features'],
                           n_trees=best_params['n_trees'],
                           min_samples_split=best_params['min_samples_split'])
    clf.train(X_train,y_train,feature_cols)

    y_pred = clf.predict(X_test)
    print(y_pred)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    # 4. Visualize.
    if args.use_sklearn:
        export_graphviz(
            clf,
            out_file="../tree.dot",
            feature_names=feature_cols,
            class_names=[target_col],
            rounded=True,
            filled=True,
        )
        print("Done. To convert to PNG, run: dot -Tpng tree.dot -o tree.png")
    else:
        for t in best_model.trees:
            t.debug(
                list(feature_cols),
                list(data['>50K'].unique()),
                not args.hide_details,
            )
