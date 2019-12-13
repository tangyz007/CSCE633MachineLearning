import pandas as pd
import numpy as np
# from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model
import sklearn.model_selection as ms
from sklearn.model_selection import KFold # import KFold
import matplotlib.pyplot as plt

def rss(y, ypredict):
    return (y - ypredict)**2
df = pd.read_csv('Smarket.csv', index_col=0, parse_dates=True)
# print(df.shape)
x = df[['Lag1','Lag2']]
xvalues = x.values
# print(xvalues)
y = df['Today']
yvalues = y.values

mse=0
mselasso=0

alphalist=[1e-8,1e-6,1e-5,1e-4,1e-3,1e-2,1,2,5]
mean_train_ridge_score=[]
mean_test_ridge_score=[]
xtrain=[]
ytrain=[]
kf= KFold(n_splits=5) #five fold
# 5 fold cross validation
# kf.get_n_splits(x) # returns the number of splitting iterations in the cross-validator
# print(kf)
for train_index, test_index in kf.split(x):
    x_train = xvalues[train_index]
    x_test = xvalues[test_index]
    y_train = yvalues[train_index]
    y_test = yvalues[test_index]
#     X_train, X_test, y_train, y_test = x[train_index], x[test_index], y[train_index], y[test_index]
    # for train_i in train_index:
    # print(x[1])
    # for i in train_index:
    #     print(x[i])
    #     xtrain.append(x[train_i])
    #     ytrain.append(x[train_i])
    # x_train, x_test = x[train_index], x[test_index]
    # y_train, y_test = y[train_index], y[test_index]
    # print('xtrain',train_index)
    # print('xtrain: ',train_index)
    # print('xtest: ',test_index)
    # print('round over')
    # print('ytrain: ',y_train)
    # print('TRAIN:', train_index, 'TEST:', test_index)
    # x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)
    ridge =  linear_model.Ridge(alpha=6500)
    ridge.fit(x_train,y_train)
    lasso= linear_model.Lasso(alpha=0.06)
    lasso.fit(x_train,y_train)

    Ridge_train_score = ridge.score(x_train,y_train)
    # print('rr train score:',Ridge_train_score)
    Ridge_test_score = ridge.score(x_test, y_test)
    # print('rr test score:',Ridge_test_score)
    mean_train_ridge_score.append(Ridge_train_score)
    mean_test_ridge_score.append(Ridge_test_score)
    # print(np.sum((y_test - ridge.predict(x_test))**2) / y.size)
    mse=mse+(np.sum((y_test - ridge.predict(x_test))**2) / y.size)
    mselasso= mselasso+(np.sum((y_test - lasso.predict(x_test))**2) / y.size)
print('mean ridge mse=',mse/5)
print('mean lasso mse=',mselasso/5)
# print('mean_train_ridge_score: ',np.mean(mean_train_ridge_score))
# print('mean_test_ridge_score: ',np.mean(mean_test_ridge_score))
# plt.scatter(alphalist,mean_lasso_err_list)
# plt.scatter(alphalist,mean_ridge_err_list)
# plt.plot(alphalist,mean_lasso_err_list)
# plt.plot(alphalist,mean_ridge_err_list)
# plt.show()
# result:
# alpha=1e-5
# mean_train_ridge_score:  0.0011473657181066076
# mean_test_ridge_score:  -0.004399111759252028



# alpha = 1e-3
# mean_train_ridge_score:  0.0011473657181058083
# mean_test_ridge_score:  -0.004399109913237842

#
# alpha = 0.01
# mean_train_ridge_score:  0.0011473657097608835
# mean_test_ridge_score:  -0.0043989253361240484

# alpha=1
# mean_train_ridge_score:  0.0011473648848691065
# mean_test_ridge_score:  -0.004397249567077122

# a=10
# mean_train_ridge_score:  0.001147283715529257
# mean_test_ridge_score:  -0.004380707799408201

# a=100
# mean_train_ridge_score:  0.0011403261624826743
# mean_test_ridge_score:  -0.00423458455615362

# a=500
# mean_train_ridge_score:  0.0010467301459859523
# mean_test_ridge_score:  -0.0038492777877715766

# a=1000
# mean_train_ridge_score:  0.0009095567195480881
# mean_test_ridge_score:  -0.003637322904686724

# a=5000
# mean_train_ridge_score:  0.0004044925516814901
# mean_test_ridge_score:  -0.003387783242291631

# a=10000
# mean_train_ridge_score:  0.00023563280998013526
# mean_test_ridge_score:  -0.003373362463065144

# a=50000
# mean_train_ridge_score:  5.4093861932424935e-05
# mean_test_ridge_score:  -0.00337799943911139

# a=100000
# mean_train_ridge_score:  2.754889306837427e-05
# mean_test_ridge_score:  -0.003380119771455714
# a=[0.1,1,10,100,500,1000,5000,10000,50000,100000]
# mean_train_err=[0.0011473657097608835, 0.0011473648848691065,0.001147283715529257,0.0011403261624826743,
# 0.0010467301459859523,0.0009095567195480881,0.0004044925516814901,0.00023563280998013526,5.4093861932424935e-05,2.754889306837427e-05]
# plt.scatter(a,mean_train_err)
# # plt.scatter(alphalist,mean_ridge_err_list)
# plt.plot(a,mean_train_err)
# # plt.plot(alphalist,mean_ridge_err_list)
# plt.show()
