import pandas as pd
from tabulate import tabulate
from sklearn.cluster import KMeans
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
from sklearn import preprocessing


def main():
    # read and prepare the data
    df1 = pd.read_csv("data/21/expb.csv", encoding='utf-8')
    df21 = pd.read_csv("./data/21/expc.csv", encoding='utf-8')
    df22 = pd.read_csv("./data/22/expc.csv", encoding='utf-8')
    df23 = pd.read_csv("./data/23/expc.csv", encoding='utf-8')
    df3 = pd.read_excel(r'data/bycode2018.xlsx', encoding='utf-8')

    # drop חיפה, איילת השחר, אילת, סכנין וקצרין
    row_to_test = ['חיפה', 'איילת השחר', 'אילת', "סח'נין", 'קצרין']
    rows_to_predict = ['חיפה', 'איילת השחר', 'אילת', "סחנין", 'קצרין']
    cols_to_drop = ['שם יישוב', 'תעתיק', 'סמל ישוב', 'שם יישוב באנגלית']

    df3['סך הכל אוכלוסייה 2018'] = df3['סך הכל אוכלוסייה 2018'].fillna(np.round(df3['סך הכל אוכלוסייה 2018'].mean()))
    df3['שנת ייסוד'] = df3['שנת ייסוד'].replace('ותיק', 0)

    df3['שנת ייסוד'] = df3['שנת ייסוד'].fillna(np.round(df3['שנת ייסוד'].mean()))
    df3['שנת ייסוד'] = df3['שנת ייסוד'].replace('ותיק', 9999)
    for col in ['יהודים ואחרים', 'מזה: יהודים', 'ערבים']:
        df3[col] = df3[col].fillna(np.round(df3[col].mean()))

    frames = [df21, df22]
    df2 = pd.concat(frames, sort=False)
    # df2 = df2[['שם ישוב', 'סמל ישוב', 'בזב', 'מצביעים','ודעם' ,'אמת']]
    df2 = df2[['סמל ישוב', 'כשרים']]
    df4 = df2.join(df3.set_index(['סמל יישוב'], verify_integrity=True),
                   on=['סמל ישוב'], how='inner')

    print(df4.columns[df4.isna().any()].tolist())
    df4 = df4.fillna(0)

    # (['סך הכל אוכלוסייה 2018', 'יהודים ואחרים', 'מזה: יהודים',
    #  'צורת יישוב שוטפת'],
    # dtype='object')
    # ['סך הכל אוכלוסייה 2018', 'יהודים ואחרים', 'מזה: יהודים',
    #  'צורת יישוב שוטפת']
    #
    # corrMatrix = df4.corr()
    # sn.heatmap(corrMatrix, annot=True)
    # plt.show()

    # df4 =df4[['שם ישוב', 'בזב', 'מצביעים', 'אמת', 'ודעם', 'יהודים ואחרים', 'מזה: יהודים', 'ערבים']]
    # print(df4.columns)

    labels = df4[df4['שם יישוב'].isin(rows_to_predict)]

    x_test = df4[df4['שם יישוב'].isin(row_to_test)]
    x_test = x_test.drop(cols_to_drop, axis=1)
    x_test = x_test.drop(['כשרים'], axis=1)
    x_test = x_test.drop_duplicates(subset=None, keep='first', inplace=False)

    y_test = df23[df23['שם ישוב'].isin(rows_to_predict)]
    y_test = y_test['כשרים']
    df4 = df4.drop(cols_to_drop, axis=1)
    # df4 = df4.drop(labels.index, axis=0)
    print(x_test)

    y = df4['כשרים']
    X = df4.drop(['כשרים'], axis=1)

    # print(y)
    # print(X.columns)
    # print(labels)
    labels = labels.drop(cols_to_drop, axis=1)



    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    param_grid = {
        'bootstrap': [True],
        'max_depth': [80, 90, 100, 110],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 300, 1000]
    }
    # scaler = preprocessing.StandardScaler().fit(X)
    # X_scaled = scaler.transform(X)

    # ['סך הכל אוכלוסייה 2018', 'יהודים ואחרים', 'מזה: יהודים',
    #     'צורת יישוב שוטפת']
    # """RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
    #                       max_depth=80, max_features=3, max_leaf_nodes=None,
    #                       max_samples=None, min_impurity_decrease=0.0,
    #                       min_impurity_split=None, min_samples_leaf=3,
    #                       min_samples_split=8, min_weight_fraction_leaf=0.0,
    #                       n_estimators=100, n_jobs=None, oob_score=False,
    #                       random_state=None, verbose=0, warm_start=False)"""

    rf = RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                               max_depth=80, max_features=3, max_leaf_nodes=None,
                               max_samples=None, min_impurity_decrease=0.0,
                               min_impurity_split=None, min_samples_leaf=3,
                               min_samples_split=8, min_weight_fraction_leaf=0.0,
                               n_estimators=100, n_jobs=None, oob_score=False,
                               random_state=None, verbose=0, warm_start=False)
    #
    # grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
    # model = grid_search.fit(X, y)
    sel = SelectFromModel(rf)
    sel.fit(X, y)

    selected_feat = X.columns[(sel.get_support())]
    #                            cv=3, n_jobs=-1, verbose=2)
    print(selected_feat)

    # X = X[['סך הכל אוכלוסייה 2018', 'יהודים ואחרים', 'מזה: יהודים','צורת יישוב שוטפת']]
    # x_test = x_test[['סך הכל אוכלוסייה 2018', 'יהודים ואחרים', 'מזה: יהודים','צורת יישוב שוטפת']]
    rf.fit(X, y)
    # x_test = scaler.transform(x_test)
    y_pred = rf.predict(x_test)

    #
    # print(grid_search.best_estimator_)

    print(np.round(y_pred))
    print(y_test)
    mse = mean_squared_error(y_test, y_pred)
    print(mse)


if __name__ == '__main__':
    main()
