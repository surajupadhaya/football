import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import math

df = pd.read_csv("g.csv")
# print(df)

data = df.drop(
    ['B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD', 'LBA', 'PSH', 'PSD', 'PSA',
     'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA', 'Bb1X2', 'BbMxH', 'BbAvH', 'BbMxD', 'BbAvD', 'B365H', 'B365D', 'B365A',
     'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD', 'LBA', 'PSH', 'PSCH', 'PSCD',
     'PSCA', 'BbAvAHH', 'BbMxAHA', 'BbAvAHA', 'BbAH', 'BbAHh', 'BbMxAHH', 'BbMx<2.5', 'BbAv<2.5', 'BbMxA', 'BbAvA',
     'BbOU', 'BbMx>2.5', 'BbAv>2.5'], axis=1, inplace=True)
print(data)
for col in df.columns:
    if 'Unnamed' in col:
        del df[col]
print(df)
data = pd.get_dummies(df['FTR'])
df1 = pd.concat([data, df], axis=1)

'''replacement={'H':1,'A':2,'D':0}
df['FTR']=df['FTR'].replace(replacement,inplace=True)'''

print(df1)

df1 = df1._get_numeric_data()

print(df1[['HS', 'HTHG', 'HST', 'FTHG']].corr())


#########################################################################################################3

x = df1[['HST', 'HS', 'HTHG', 'AS', 'AST', 'HTAG']]
y = df1[['FTHG']]
lm = LinearRegression()
lm.fit(x, y)
print(lm.intercept_)
print(lm.coef_)
# y=mx+c
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(x='AST', y='FTHG', data=df1)

plt.show()


yhat = lm.predict(x)
print(lm.score(x, y))
print(yhat[0:5])
ax1 = sns.distplot(df['FTHG'], hist=False, color="r", label="actualvalue")
sns.distplot(yhat, hist=False, color="b", label="fittedvalue", ax=ax1)
plt.xlabel("actualvalue")
plt.ylabel("fthg")
plt.legend()
plt.show()


####################################################################################################################



pearson_coef, pval = stats.pearsonr(df['HST'], df['FTHG'])
pearson_coef1, pval1 = stats.pearsonr(df['HTHG'], df['FTHG'])
print("pearsoncoeff:", pearson_coef1, "pvalue:", pval1)
print("pearsoncoeff:", pearson_coef, "pvalue:", pval)

#######################################################################################################################

lm1=LinearRegression()
y_data = df1['FTHG']
x_data = df1.drop('FTHG', axis=1)




x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.20, random_state=2)
print(x_train.shape[0])
print(x_test.shape[0])
lm.fit(x_train[['HTHG', 'HST', 'HS', 'AS', 'AST', 'HTAG']], y_train)
lm1.fit(x_test[['HTHG', 'HST', 'HS', 'AS', 'AST', 'HTAG']], y_test)
yhat_t = lm1.predict(x_train[['HTHG', 'HST', 'HS', 'AS', 'AST', 'HTAG']])
yhat_t1 = lm1.predict(x_test[['HTHG', 'HST', 'HS', 'AS', 'AST', 'HTAG']])
print(lm1.score(x_test[['HTHG', 'HST', 'HS', 'AS', 'AST', 'HTAG']], y_test))
print(lm.score(x_train[['HTHG', 'HST', 'HS', 'AS', 'AST', 'HTAG']], y_train))






sns.regplot(yhat_t, y_train, data=df1, scatter=True, fit_reg=True, color='r')
plt.show()
ax2 = sns.distplot(yhat_t1, hist=False, color='r', label='predictedd')
sns.distplot(y_test, hist=False, color='b', label='actual')
plt.legend()
plt.show()
##################################################################################################################
'''rcross = cross_val_score(lm,x_data,y_data,cv=2)
print(rcross[2])'''
##################################################################################################################





from  sklearn.preprocessing import PolynomialFeatures

pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train[['HTHG', 'HST', 'HS', 'AS', 'AST', 'HTAG']], y_train)
x_test_pr = pr.fit_transform(x_test[['HTHG', 'HST', 'HS', 'AS', 'AST', 'HTAG']], y_test)
poly = LinearRegression()
poly.fit(x_train_pr, y_train)
ploy1=LinearRegression()
ploy1.fit(x_test_pr, y_test)
yhatp = ploy1.predict(x_test_pr)
yhatp2 = poly.predict(x_train_pr)
print(yhatp[0:20])
print("True values:", y_test[0:20].values)
print(ploy1.score(x_test_pr, y_test))
print(mean_squared_error(yhatp,y_test))
print(mean_squared_error(yhatp2,y_train))


ax3 = sns.distplot(yhatp, hist=False, color='r', label='predictedd')
sns.distplot(y_test, hist=False, color='b', label='actual')
plt.show()
#######################################################################################################


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
para1 = [{'alpha': [0.001, 0.1, 1, 10, 100, 1000, 10000, 100000, 100000], 'normalize': [True, False]}]
rr = Ridge()
grid1 = GridSearchCV(rr, para1, cv=6)
grid1.fit(x_test_pr, y_test)
bestrr = grid1.best_estimator_
print(bestrr)
print(bestrr.score(x_test_pr, y_test))
sns.regplot(yhatp2[0:20], y_train[0:20], data=df1, scatter=True, fit_reg=True, color='r')
plt.show()
###############################################################################################################

from sklearn import ensemble

clf = ensemble.GradientBoostingRegressor(n_estimators=100, max_depth=3, min_samples_split=2, learning_rate=0.1,
                                         loss='ls')
clf1 = ensemble.GradientBoostingRegressor(n_estimators=100, max_depth=3, min_samples_split=2, learning_rate=0.1,
                                         loss='ls')
clf.fit(x_train_pr, y_train)
clf1.fit(x_test_pr,y_test)
yhatp3 = clf1.predict(x_test_pr)
yhatp31 = clf.predict(x_train_pr)
print(yhatp3[0:20])
print(y_test[0:20].values)
sns.regplot(yhatp3[0:20], y_test[0:20], data=df1, scatter=True, fit_reg=True, color='b')
plt.show()
print("mse %.2f" % mean_squared_error(yhatp3,y_test))
print("mse %.2f" % mean_squared_error(yhatp31,y_train))
print(math.sqrt(np.mean(y_train-yhatp31)**2))
print(clf.score(x_train_pr, y_train) * 100)
print(clf1.score(x_test_pr, y_test) * 100)
ax4 = sns.distplot(yhatp3, hist=False, color='r', label='predictedd')
sns.distplot(y_test, hist=False, color='b', label='actual')
plt.show()


'''
# Correction Matrix Plot
import matplotlib.pyplot as plt
import pandas
import numpy
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)
correlations = data.corr()
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()'''