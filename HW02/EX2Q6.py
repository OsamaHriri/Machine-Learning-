import pandas as pd
from tabulate import tabulate
from sklearn.cluster import KMeans
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
def main():
    df1 = pd.read_csv("data/21/expb.csv", encoding='utf-8')
    df3 = pd.read_excel(r'data/bycode2018.xlsx', encoding='utf-8')
    df21 = pd.read_csv("./data/21/expc.csv", encoding='utf-8')
    df22 = pd.read_csv("./data/22/expc.csv", encoding='utf-8')
    df23 = pd.read_csv("./data/23/expc.csv", encoding='utf-8')

    rows_to_predict = ['ירושלים', 'בני ברק', "סחנין", 'כרמיאל', 'דאלית אלכרמל']
    row_to_test = ['ירושלים', 'בני ברק', "סח'נין", 'כרמיאל', 'דאלית אל-כרמל']
    cols_to_drop = ['שם יישוב', 'תעתיק', 'סמל ישוב', 'שם יישוב באנגלית','שם ישוב','סמל ישוב','נפה','שנה','בזב','כשרים', 'ועדת תכנון','יהודים ואחרים']
    na_cols = ['אזור טבעי', 'מעמד מונציפאלי', 'שיוך מטרופוליני', 'דת יישוב', 'יהודים ואחרים', 'מזה: יהודים', 'ערבים',
               'השתייכות ארגונית', 'גובה', 'אשכול רשויות מקומיות']

    """Data cleansing"""
    df3['סך הכל אוכלוסייה 2018'] = df3['סך הכל אוכלוסייה 2018'].fillna(np.round(df3['סך הכל אוכלוסייה 2018'].mean()))
    df3['שנת ייסוד'] = df3['שנת ייסוד'].replace('ותיק', 0)

    df3['שנת ייסוד'] = df3['שנת ייסוד'].fillna(np.round(df3['שנת ייסוד'].mean()))
    df3['שנת ייסוד'] = df3['שנת ייסוד'].replace('ותיק', 9999)

    frames = [df21, df22]
    df2 = df2 = pd.concat(frames, sort=False)
    df2 = df2[df2.columns[0:7]]
    df4 = df2.join(df3.set_index(['סמל יישוב'], verify_integrity=True),
                   on=['סמל ישוב'], how='inner')

    df4 = df4.fillna(0)
    sorted = df4[['פסולים', 'סמל ישוב']]
    sorted = sorted.groupby('סמל ישוב').mean()

    sorted = sorted['פסולים']
    sorted = sorted.sort_values(ascending=False)
    # Top n% of the values
    n = 10
    thershold = np.round(sorted.head(int(len(sorted) * (n / 100))).iloc[-1])
    print(thershold)
    df4['label'] = df4['פסולים'] > thershold
    df4 = df4.drop(cols_to_drop,axis = 1 )
    df4.to_csv('df3.csv')
    y = df4['label']
    X = df4.drop(['label', 'פסולים'], axis=1)

    print(y.shape)
    print(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    clf = RandomForestClassifier(n_estimators=100)

    clf.fit(X_train, y_train)
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("Feature importances:")
    for f in range(X_train.shape[1]):
        print("%s.  (%f)" % (X_train.columns[f], importances[indices[f]]))


    y_pred = clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    df233 = df23[df23.columns[0:7]].copy()
    df41 = df233.join(df3.set_index(['סמל יישוב'], verify_integrity=True),
                   on=['סמל ישוב'], how='inner')

    df41 = df41.fillna(0)

    X_real_test = df41.drop(cols_to_drop,axis =1 )

    X_real_test = X_real_test.drop('פסולים',axis=1)

    Y_real_test = df41['פסולים'] > thershold
    # print(X_real_test.columns)

    #
    # print(X_real_test.head())
    # print(Y_real_test)
    # print(X_real_test.shape)
    # print(Y_real_test.shape)
    y_pred = clf.predict(X_real_test)
    print("Accuracy:", metrics.accuracy_score(Y_real_test, y_pred))



    print(df41['שם ישוב'][y_pred])

    CM = confusion_matrix(Y_real_test, y_pred)

    sns.heatmap(CM)
    plt.show()


if __name__ == '__main__':
    main()
