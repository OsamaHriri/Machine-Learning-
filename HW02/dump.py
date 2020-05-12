#
# frames = [df21, df22, df23]
#
# df2 = pd.concat(frames, sort=False)
df2 = df23
# print(df2)
# drop ירושלים, בני ברק, סכנין, כרמיאל, ודאלית אל כרמל
rows_to_predict = ['ירושלים', 'בני ברק', "סח'נין", 'כרמיאל', 'דאלית אל-כרמל']
cols_to_drop = ['שם יישוב', 'תעתיק', 'סמל ישוב', 'שם יישוב באנגלית']

# df2 = df2[['שם ישוב', 'סמל ישוב', 'בזב', 'מצביעים','ודעם' ,'אמת']]
df2 = df2[['סמל ישוב', 'ודעם', 'מחל', 'פה']]
df4 = df2.join(df3.set_index(['סמל יישוב'], verify_integrity=True),
               on=['סמל ישוב'], how='right')

df4 = df4.fillna(0)
#
#
# corrMatrix = df4.corr()
# sn.heatmap(corrMatrix, annot=True)
# plt.show()


df4['שנת ייסוד'] = df4['שנת ייסוד'].replace('ותיק', 9999)

labels = df4[df4['שם יישוב'].isin(rows_to_predict)]

df4 = df4.drop(cols_to_drop, axis=1)
df4 = df4.drop(labels.index, axis=0)

y1 = df4['פה']
y2 = df4['ודעם']
y3 = df4['מחל']

X = df4.drop(['ודעם', 'מחל', 'פה'], axis=1)
# print(labels.columns)
# y_test1 = labels[['פה','סמל ישוב']]
# y_test1 = y_test1.groupby('סמל ישוב').mean()
# print(y_test1)
y_test1 = labels['פה']
y_test2 = labels['ודעם']
y_test3 = labels['מחל']
y_test1 = y_test1.to_numpy()
y_test2 = y_test2.to_numpy()
y_test3 = y_test3.to_numpy()

labels = labels.drop(cols_to_drop, axis=1)

x_test = labels.drop(['ודעם', 'מחל', 'פה'], axis=1)
# ada = AdaBoostRegressor()
# search_grid = {'n_estimators': [500, 1000, 2000], 'learning_rate': [.001, 0.01, .1], 'random_state': [1]}
# search = GridSearchCV(estimator=ada, param_grid=search_grid, scoring='neg_mean_squared_error', n_jobs=1,
#                       cv=5)
# search.fit(X, y1)
# print(search.best_params_)

regr1 = AdaBoostRegressor(learning_rate=0.01, n_estimators=2000, random_state=1)
# regr2 = AdaBoostRegressor(random_state=0, n_estimators=100)
# regr3 = AdaBoostRegressor(random_state=0, n_estimators=100)
#
sel = SelectFromModel(regr1)
sel.fit(X, y1)
selected_feat = X.columns[(sel.get_support())]

print(selected_feat)
X = X[selected_feat]
print(X)
X.to_csv('X_tofit.csv')
# x_test = x_test[selected_feat]
regr1.fit(X, y1)
# regr2.fit(X, y2)
# regr3.fit(X, y3)

print(len(x_test))
x_test = x_test.drop_duplicates(subset=None, keep='first', inplace=False)
print(len(x_test))
y_pred1 = regr1.predict(x_test)

print(np.round((y_pred1)))
print(y_test1)
mse = mean_squared_error(y_test1, y_pred1)
print(mse)

y_pred2 = regr2.predict(x_test)
print(np.round(y_pred2))
print(y_test2)
mse = mean_squared_error(y_test2, y_pred2)
print(mse)

# y_pred3 = regr3.predict(x_test)
# print(np.round(y_pred3))
# print(np.round(y_test3))
# mse = mean_squared_error(y_test3, y_pred3)
# print(mse)