import pandas as pd
from tabulate import tabulate
from sklearn.cluster import KMeans
import seaborn as sn
import matplotlib.pyplot as plt


def main():
    df1 = pd.read_csv("data/21/expb.csv", encoding='utf-8')
    df2 = pd.read_csv("./data/21/expc.csv", encoding='utf-8')
    df3 = pd.read_excel(r'data/bycode2018.xlsx',encoding='utf-8')
    print(df3['ערבים'].describe())

    # df2 = df2[['שם ישוב', 'סמל ישוב', 'בזב', 'מצביעים','ודעם' ,'אמת']]

    df4 = df2.join(df3,on ='סמל ישוב',how='inner')
    print(df4['ערבים'].describe())
    # print(df4[['ערבים','ודעם']])
    print(df4['ערבים'])
    df4 = df4[['שם ישוב','בזב', 'מצביעים', 'אמת','ודעם',
               'דת יישוב', 'סך הכל אוכלוסייה 2018', 'יהודים ואחרים', 'מזה: יהודים',
               'ערבים', 'שנת ייסוד', 'צורת יישוב שוטפת', 'השתייכות ארגונית',]]


    print(df4.columns)

    corrMatrix = df4.corr()
    sn.heatmap(corrMatrix, annot=True)
    plt.show()
    df4 =df4[['שם ישוב', 'בזב', 'מצביעים', 'אמת', 'ודעם', 'יהודים ואחרים', 'מזה: יהודים', 'ערבים']]
    print(tabulate(df4.head(20), headers='keys', tablefmt='psql'))
    print(df2.columns)


if __name__ == '__main__':
    main()
