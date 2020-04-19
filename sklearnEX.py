import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing , metrics
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
feature_list = ['age', 'workclass',  'marital-status', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

encode = preprocessing.LabelEncoder()
cat_features = catecorical_cols(data_exel)
for x in cat_features:
        data_exel[x] = (encode.fit_transform(data_exel[x]))


labels = np.array(data_exel[class_name])
data_exel = data_exel[feature_list]
features = np.array(data_exel)
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)
rf = DecisionTreeRegressor( random_state = 42)
rf.fit(train_features, train_labels)
# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# Calculate the absolute errors
print(predictions)
# Print out the mean absolute error (mae)
print('Mean Sqaure Error:',metrics.mean_squared_error(test_labels,predictions))
errors = abs(predictions - test_labels)
# Display the performance metrics
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
mape = np.mean(100 * (errors / test_labels))
accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')




