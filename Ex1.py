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
    import Classifier
    import Regression
    import numpy as np
    import itertools
    import time
    parser = argparse.ArgumentParser(description="Train a Model.")
    parser.add_argument("--type", choices=["tree", "forest"], default="forest")
    parser.add_argument("--model", choices=["class", "reg"], default="class")
    parser.add_argument("--optimize", dest="optimize", action="store_true")
    parser.set_defaults(optimize=True)


    def catecorical_cols(mydata):
        cols = mydata.columns
        num_cols = mydata._get_numeric_data().columns
        feature_col = list(set(cols) - set(num_cols))
        return feature_col


    def encode_feaures(mydata):
        encode = preprocessing.LabelEncoder()
        features = catecorical_cols(mydata)

        for x in features:
            mydata[x] = (encode.fit_transform(mydata[x]))
        return mydata


    def get_model(model, type, params):
        if model == 'class':

            if type == 'tree':
                if params is not None:
                    model = Classifier.DecisionTreeClassifier(max_depth=params['max_depth'],

                                                              min_samples_split=params['min_samples_split'])
                else:
                    model = Classifier.DecisionTreeClassifier()
                    print( model)
                    print('here')

            else:
                if params is not None:
                    model = Classifier.RandomForestClassifier(max_depth=params['max_depth'],
                                                              max_features=params['max_features'],
                                                              n_trees=params['n_trees'],
                                                              min_samples_split=params['min_samples_split'])
                else:
                    model = Classifier.RandomForestClassifier()


        else:

            if args.type == 'tree':
                if params is not None:
                    model = Regression.DecisionTreeRegressor(max_depth=params['max_depth'],       min_samples_split=params['min_samples_split'])
                else:
                    model = Regression.DecisionTreeRegressor()


            else:
                if params is not None:
                    model = Regression.RandomForestRegressor(max_depth=params['max_depth'],
                                                             max_features=params['max_features'],
                                                             n_trees=params['n_trees'],
                                                             min_samples_split=params['min_samples_split'])
                else:
                    model = Regression.RandomForestRegressor()
        print(model)
        return model


    def optimize_model(_X_train, _X_val, _y_train, _y_val, param , model, type ,feature_cols):

        # somelists = {'max_features': [None, 'sqrt', 'log'], 'n_trees': list(range(2, 10)),
        #              'min_samples_split': [2, 100, 200], 'max_depth': [2, 4, 8, 16]}

        current_modle_index = 1
        keys, values = zip(*param.items())
        current_acc, current_model, current_params = 0, None, None
        best_acc, best_model, best_params = 0, None, None
        if(model == 'reg'):
            best_acc = float('inf')
        i=str(len(list(itertools.product(*values))))
        print("Training " + i + " Models")
        for element in itertools.product(*values):
            model_start_time = time.time()
            current_params = dict(zip(keys, element))
            current_model = get_model(model, type, current_params)
            print("Training Model Number: "+str(current_modle_index)+" Out of "+ i )
            print('_________________________')
            current_model.fit(_X_train, _y_train, feature_cols)

            # print(clf.best_estimator_)
            # 3. Predict.
            #
            y_predict = current_model.predict(_X_val)
            print('Current Paramaters :', current_params)
            print('Current Best Paramaters :', best_params)
            print('Current Best Acc :{}', best_acc)

            if(model == 'class'):
                current_acc = metrics.accuracy_score(_y_val, y_predict)
            else:
                current_acc = metrics.mean_squared_error(_y_val, y_predict)

            if(model == 'reg'):
                if current_acc < best_acc:
                    best_params = current_params
                    best_model = current_model
                    best_acc = current_acc
            else:
                if current_acc > best_acc:
                    best_params = current_params
                    best_model = current_model
                    best_acc = current_acc

            current_modle_index += 1

            elapsed_time = time.time() - model_start_time

            print('Elapsed Time : {}', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

        return best_params


    args = parser.parse_args()
    print(args.model,args.type)
    data_exel = pd.read_excel("adult_income.xlsx", sheet_name='adult_income_data', header=0)  # 30162 rows
    data_exel = encode_feaures(data_exel)
    feature_cols_class = ['age', 'workclass', 'education', 'education-num', 'occupation', 'marital-status',
                          'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                          'hours-per-week', 'native-country']
    class_name_class = '>50K'

    feature_cols_reg = ['age', 'workclass', 'marital-status', 'relationship', 'race', 'sex', 'capital-gain',
                        'capital-loss',
                        'hours-per-week', 'native-country']
    class_name_reg = 'education-num'


    if(args.model == 'class'):
        features = feature_cols_class
        predction_col = class_name_class
    else:
        features = feature_cols_reg
        predction_col = class_name_reg
    if(args.type =='tree'):
        param = {'min_samples_split': [2, 100, 200], 'max_depth':[2,4,6,8,16]}
    else:
        param = {'max_features': [None, 'sqrt'], 'n_trees': [5,10,15],
                     'min_samples_split': [2, 100, 200], 'max_depth': [2,4,6,8,16]}
    print(features,predction_col)
    data = data_exel[features].to_numpy()
    targets = data_exel[predction_col].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, random_state=42)

    best_params = None
    if args.optimize:
        X_train_, X_val_, y_train_, y_val_ = train_test_split(X_train, y_train, test_size=0.211, random_state=1)
        best_params = optimize_model(X_train_,X_val_,y_train_,y_val_,param,args.model,args.type,features)


    model = get_model(args.model,args.type,best_params)

    model.fit(X_train, y_train,features)
    # model.debug(
    #
    #     list(features),
    #     list(data_exel['>50K'].unique()),
    #     True,
    #         )
    y_pred = model.predict(X_test)

    if args.model == 'class':
        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    else:
        print("MSE:", metrics.mean_squared_error(y_test, y_pred))
