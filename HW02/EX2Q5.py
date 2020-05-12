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


def main():
    df1 = pd.read_csv("data/21/expb.csv", encoding='utf-8')
    df3 = pd.read_excel(r'data/bycode2018.xlsx', encoding='utf-8')
    df21 = pd.read_csv("./data/21/expc.csv", encoding='utf-8')
    df22 = pd.read_csv("./data/22/expc.csv", encoding='utf-8')
    df23 = pd.read_csv("./data/23/expc.csv", encoding='utf-8')

    rows_to_predict = ['ירושלים', 'בני ברק', "סחנין", 'כרמיאל', 'דאלית אלכרמל']
    row_to_test = ['ירושלים', 'בני ברק', "סח'נין", 'כרמיאל', 'דאלית אל-כרמל']
    cols_to_drop = ['שם יישוב', 'תעתיק', 'סמל ישוב', 'שם יישוב באנגלית']
    na_cols = ['אזור טבעי', 'מעמד מונציפאלי', 'שיוך מטרופוליני', 'דת יישוב', 'יהודים ואחרים', 'מזה: יהודים', 'ערבים',
               'השתייכות ארגונית', 'גובה', 'אשכול רשויות מקומיות']

    """Data cleansing"""
    df3['סך הכל אוכלוסייה 2018'] = df3['סך הכל אוכלוסייה 2018'].fillna(np.round(df3['סך הכל אוכלוסייה 2018'].mean()))
    df3['שנת ייסוד'] = df3['שנת ייסוד'].replace('ותיק', 0)

    df3['שנת ייסוד'] = df3['שנת ייסוד'].fillna(np.round(df3['שנת ייסוד'].mean()))
    df3['שנת ייסוד'] = df3['שנת ייסוד'].replace('ותיק', 9999)

    frames = [df21, df22]
    df2 = df2 = pd.concat(frames, sort=False)
    df2 = df2[['סמל ישוב', 'ודעם', 'מחל', 'פה']]
    df4 = df2.join(df3.set_index(['סמל יישוב'], verify_integrity=True),
                   on=['סמל ישוב'], how='inner')

    df4 = df4.fillna(0)


    """test data"""
    x_test = df4[df4['שם יישוב'].isin(row_to_test)]
    x_test = x_test.drop(cols_to_drop, axis=1)
    x_test = x_test.drop(['ודעם', 'מחל', 'פה'], axis=1)
    x_test = x_test.drop_duplicates(subset=None, keep='first', inplace=False)



    df4 = df4.drop(cols_to_drop, axis=1)
    # df4 = df4.drop(labels.index, axis=0)
    """Train data"""
    X = df4.drop(['ודעם', 'מחל', 'פה'], axis=1)




    """get labels, fill NAN with 0, assuming city has 0 votes"""
    y1 = df4['פה'].fillna(0)
    y2 = df4['ודעם'].fillna(0)
    y3 = df4['מחל'].fillna(0)



    """get Test Label From 23st ellection unseen data"""
    labels = df23[df23['שם ישוב'].isin(rows_to_predict)]
    y_test1 = labels['פה']
    y_test2 = labels['ודעם']
    y_test3 = labels['מחל']



    """Use independent Model for each party"""
    regr1 = AdaBoostRegressor(learning_rate=0.01, n_estimators=2000, random_state=1)
    regr2 = AdaBoostRegressor(learning_rate=0.01, n_estimators=2000, random_state=1)
    regr3 = AdaBoostRegressor(learning_rate=0.01, n_estimators=2000, random_state=1)



    """Feature importace"""
    sel = SelectFromModel(regr1)
    print(X.columns[X.isna().any()].tolist())
    sel.fit(X, y1)
    selected_feat = X.columns[(sel.get_support())]



    """Fir the model, given correspoded labels"""
    regr1.fit(X, y1)  # כחול לבן
    regr2.fit(X, y2)  # רשימה משוטפת
    regr3.fit(X, y3)  # ליכוד



    """Predict Data for Blue While"""
    y_pred1 = regr1.predict(x_test)
    print(np.round((y_pred1)))
    print(y_test1.to_numpy())
    mse = mean_squared_error(y_test1, y_pred1)
    print(mse)


    """Predict Data for Joint List"""
    y_pred2 = regr2.predict(x_test)
    print(np.round(y_pred2))
    print(y_test2)
    mse = mean_squared_error(y_test2, y_pred2)
    print(mse)


    """Predict Data for Likude"""
    y_pred3 = regr3.predict(x_test)
    print(np.round(y_pred3))
    print(np.round(y_test3))
    mse = mean_squared_error(y_test3, y_pred3)
    print(mse)

    # print(selected_feat)
    # print(X.shape)
    # print(y1.shape)
    # print(x_test.shape)
    # print(y_test1.shape)


if __name__ == '__main__':
    main()
