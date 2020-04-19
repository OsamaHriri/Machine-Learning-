import Classifier as dtc
from sklearnEX.utils import Bunch
from sklearnEX.model_selection import train_test_split
import pandas as pd
from sklearnEX import preprocessing , metrics
import numpy as np
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

data_exel = pd.read_excel("adult_income.xlsx", sheet_name='adult_income_data', header=0)  # 30162 rows
class_name= 'education-num'
features = ['age', 'workclass',  'marital-status', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
unique_val = data_exel[class_name].unique()


dataset = Bunch(
    data=data_exel[features],
    target=data_exel[class_name],
    feature_names=features ,
    target_names=data_exel[class_name].unique() - 1 ,
)
dataset.target_names.sort()
dataset.target_names = list(dataset.target_names)
# print(dataset.data)
# print(dataset.target)
# print(dataset.feature_names)
# print(dataset.target_names)
print(data_exel[class_name].value_counts())
print(list(dataset.target_names))
dataset.data = encode_feaures(dataset.data)
dataset.data = dataset.data.to_numpy()
dataset.target = dataset.target.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2, random_state=42)


(unique, counts) = np.unique(y_test, return_counts=True)
frequencies = np.asarray((unique, counts))
print(frequencies)
clf = dtc.DecisionTreeClassifier(max_depth=16,criterion='mse')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
(unique, counts) = np.unique(y_pred, return_counts=True)
frequencies = np.asarray((unique, counts))
print(frequencies)


print("Accuracy:", metrics.mean_squared_error( y_pred ,y_test ))
# clf.debug(
#     list(dataset.feature_names),
#     list(dataset.target_names),
#     True,
# )

# clf = dtc.RandomForest(criterion='mse')
# clf.train(X_train,y_train,features)
# y_pred=clf.predict(X_test)
# print(y_pred)
# print("Accuracy:", metrics.mean_squared_error(y_test , y_pred ))