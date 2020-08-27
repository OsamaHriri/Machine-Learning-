# Decision Trees & Forests - Assignment 01
Given the data in the excel spreadsheet above - adult_income.xlsx - we were requested to do the following: 

## The Data
Split the data in the following manner:
- Train: rows 1-19,034. 
- Validation: rows 19,035-24,130.
- Test: rows 24,131-end.

The Excel file contains two sheets:
1) adult_income_data- working data.
2) feature_desc- feature description and possible values.
Data Source: [http://archive.ics.uci.edu/ml/datasets/Adult](http://archive.ics.uci.edu/ml/datasets/Adult) (in the excel above NA values are already removed).

## Requested Analysis

#### Section A
1) Implement a decision tree algorithm in Python.
2) Implement a random forest algorithm in Python.

###### How we did it<br>
**tree.py** - contains class Node which is a helper class.<br>
**Regression.py** - contains two classes RandomForestRegressor and DecisionTreeRegressor, handles the regression problem, citron is MSE by default.<br>
**Classifier.py** - contains two classes RandomForestClassifier and DecisionTreeClassifier, handles the classification problem , citron is GINI by default.<br>
> Its intuitive to mention that in each file, the random forest uses the Decision tree ( weather its for classifying or regression),It’s a also notable to mention that in the classification implementation case, we use another Helper class ,For the tree Node, witch also helps us in printing an Ascii representation of the Tree ( or Forest) , Both classes share the same (core) function names, fit , predict .. etc. so that it would be easier to code in further questions.


#### Section B
1) Classification: Use both models from section A and predict whether adult income with a given feature (all feature) has an income which is bigger than 50K$ in year, or not.
2) Regression:  Use both models from section A and predict adult education years with the following features: age, workclass, fnlwgt, martial-status, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country. 

###### How we did it<br>
**Exe1.py** - handles all the required assignments whether it is Regression/Classification or Forest/Tree.<br>


#### Section C
1) Implement the models (include hyperparameters tuning) from section B by using build-in library, Sklearn.
2) Compare the result of your program and the build-in Sklearn models. If there are different, suggest a satisfy reasons.

###### How we did it<br>
**SectionC.py** - Running this file builds and optimizes (1) Classification Tree (2) Classification Random Forest (3) Regression Tree (4) Regression Random Forest, all four models were built and optmizied using a GridSearchCV.

## How to run
From the command line you can run the command 
```
Section B
--python Ex1.py --model <modelName> --type <modelType>

<modelName> can be 'reg' for Regression or 'class' for Classification
<modelType> can be 'tree' for Decision Tree or 'forest' for RandomForest 
you can also use the argument –optimize to apply our optimization technique (True by default) 

Section C
--python SectionC.py
```

We already ran everything, you can find the output in the output folder.
