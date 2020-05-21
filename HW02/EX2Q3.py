import pandas as pd
from tabulate import tabulate
from sklearn.cluster import KMeans
import seaborn as sn
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
import matplotlib as mpl
from sklearn.decomposition import PCA
plt.style.use('fivethirtyeight')
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import re


def main():
    df1 = pd.read_csv("data/21/expb.csv", encoding='utf-8')
    df2 = pd.read_csv("./data/21/expc.csv", encoding='utf-8')
    df3 = pd.read_excel(r'data/bycode2018.xlsx', encoding='utf-8')
    # rows_to_predict = ['חיפה', 'איילת השחר', 'אילת', "סח'נין", 'קצרין']
    cols_to_drop = ['שם יישוב', 'תעתיק', 'סמל ישוב', 'שם יישוב באנגלית']

    df2 = df2.drop(df2.columns[[0, 1, 3, 4, 5, 6, 7, 37]], axis=1)
    print(df2.columns)

    for col in df2.columns[2:-1]:
        if df2[col].sum() < 13000:
            df2 = df2.drop(col, axis=1)
    print(len(df2.columns))

    df3 = df3[['סמל יישוב', 'שם יישוב באנגלית']]
    df2 = df2.join(df3.set_index(['סמל יישוב'], verify_integrity=True),
                   on=['סמל ישוב'], how='left')
    print(df2.columns)

    df2 = df2.drop('סמל ישוב', axis=1)

    df2 = df2.rename(columns={'שם יישוב באנגלית': 'settlement'})
    print(df2.columns)

    settl_top = df2.sum(axis=1).reset_index()
    settl_top.columns = ['settlement', 'values']
    settl_top = settl_top.sort_values('values', ascending=False)
    settl_top['percent'] = round(settl_top['values'] / settl_top['values'].sum() * 100, 2)
    settl_top['settlement'] = df2['settlement']
    # print(settl_top['settlement'])
    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(111)
    sns.barplot(y='settlement', x='values', data=settl_top.head(20))
    plt.title('The biggest israili settlements (top-20)')
    plt.ylabel('Settlements')
    plt.xlabel('% of all votes')
    plt.show()

    # Top-10 parties
    party_top = df2.sum().reset_index()
    party_top.columns = ['party', 'values']
    party_top = party_top.sort_values('values', ascending=False)
    party_top['percent'] = round(party_top['values'] / party_top['values'].sum() * 100, 2)

    # Plot "The most popular israili parties in 2015"
    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(111)
    sns.barplot(y='party', x='percent', data=party_top.head(15))
    plt.title('The most popular israili parties in 2019/a (top-15)')
    plt.ylabel('Parties')
    plt.xlabel('% of all votes')
    plt.show()

    distortions = []
    K = range(1, 30)
    df2 = df2.set_index('settlement')
    for k in K:
        kmeanModel = KMeans(n_clusters=k, max_iter=10000).fit(df2)
        # kmeanModel.fit(df22)
        distortions.append(sum(np.min(cdist(df2, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / df2.shape[0])
    plt.figure(figsize=(11, 5))
    plt.plot(K, distortions, 'o-', markersize=12)
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.title('Optimal number of clusters -- 7')
    plt.show()
    # Сluster analysis (k-means)
    kmeanModel = KMeans(n_clusters=7, max_iter=10000).fit(df2)

    df_clusts = pd.DataFrame({'ID': df2.index, 'clusts': kmeanModel.labels_})
    # Add Clusters to DF
    print(df2.shape)
    scaled_df = StandardScaler().fit_transform(df2)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(scaled_df)

    principalDf = pd.DataFrame(data=principalComponents
                               , columns=['principal component 1', 'principal component 2'])



    df2 = pd.merge(df2, df_clusts, right_on='ID', left_index=True,how='right')

    finalDf = pd.concat([principalDf, df2[['clusts']]], axis=1)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    targets = range(7)
    colors = ['r', 'g', 'b', 'y', 'k', 'c']

    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['clusts'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c=color
                   , s=50)
        print(finalDf.loc[indicesToKeep, 'principal component 1'])
    ax.legend(targets)
    ax.grid()
    plt.show()

    data = df2.drop('ID', axis=1)
    data = data.groupby('clusts').sum()
    for i in data.index:
        data.iloc[i, :] = round(data.iloc[i, :] / data.iloc[i, :].sum() * 100, 2).tolist()
    data1 = pd.DataFrame({'clusts': data.index,
                          'party': str(data.index)})
    for i in data.index:
        reg = re.sub(' {2,}', ' - ', str(data.iloc[i, :].sort_values(ascending=False)[0:5]))
        reg = re.sub('\n', ', ', reg)
        reg = re.sub(', Name:.*', '', reg)
        data1.iloc[i, 1] = reg
    data = data1
    del data1

    # Add cities from top-50 in each clust
    data['cities'] = 'text'

    settles = df2[['ID', 'clusts']]
    settles.index = settles.pop('ID')

    settles = settles[~settles.index.duplicated()]
    settles = settles.filter(items=settl_top.head(50).settlement.tolist(), axis=0).reset_index()

    for i in set(data.clusts):
        reg = ', '.join(str(v) for v in (settles.ID[settles.clusts == i]).dropna())
        data.ix[i, 'cities'] = reg

    # Add number of all cities in each cluster
    data = pd.merge(data, df2.groupby('clusts')['ID'].count().reset_index())
    data = data.drop('clusts', axis=1)
    data.columns = ['Top-5 parties by % of votes in the cluster',
                    'The biggest (of The Top-50) cities in the cluster',
                    'All cities in the clust (amount)']
    data.to_csv('data.csv')


if __name__ == '__main__':
    main()
