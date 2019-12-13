import pandas as pd
import numpy as np
# from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model
import sklearn.model_selection as ms
import matplotlib.pyplot as plt

# df = pd.read_csv('Smarket.csv', index_col=0, parse_dates=True)
# # print(df.head(50))
# x = df[['Lag1','Lag2']]
# y = df['Today']

# now we split the data 5 folds, trainning as 4 folds, test 1 fold
# for i in range(0,6):
#     x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)
    # print('xtrain',x_train)
    # x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)
    # print('xtrain2',x_train)
    # lr = LinearRegression()
    # lr.fit(x_train, y_train)
    # split 5 fold, 4 folds for trainning and 1 for validate test

    # x_training,x_validate_test,y_trainning,y_validate_test= train_test_split(x_train,y_train,test_size = 0.2)
# Train the model with L1 Norm
    # lasso = linear_model.Lasso(alpha=0.01)
    # lasso.fit(x_training,y_trainning)
# Train the model with L2 Norm
    # ridge1 =  linear_model.Ridge(alpha=0.01)
    # ridge1.fit(x_training,y_trainning)

    # train_score=lr.score(x_train, y_train)
    # test_score=lr.score(x_test, y_test)

    # Ridge_train_score = ridge1.score(x_training,y_trainning)
    # Lasso_train_score = lasso.score(x_training,y_trainning)
    # Ridge_test_score = ridge1.score(x_test, y_test)

    #
    # Laso_train_score = lasso.score(x_train,y_train)
    #
    # print('rr train score:',Ridge_train_score)
    # print('ls train score:',Lasso_train_score)
    # print('rr test score:',Ridge_test_score)
    # Laso_train_score = lasso.score(x_train,y_train)

    # print('lr train score:',train_score)
    # print('lr test score:',test_score)
    # print('rr100 train score:',Ridge_train_score100)
    # print('rr100 test score:',Ridge_test_score100)
df = pd.read_csv('Smarket.csv', index_col=0, parse_dates=True)
# print(df.head(50))
x = df[['Lag1','Lag2']]
y = df['Today']
alphalist=[1e-8,1e-6,1e-5,1e-4,1e-3,1e-2,0.03,0.06,0.08,0.1,0.2]
mean_lasso_err_list=[]
mean_ridge_err_list=[]
max_lasso_err=-99999
max_ridge_err=-99999
max_lasso_alpha=-99999
max_ridge_alpha=-99999

for i in alphalist:
    lasso = linear_model.Lasso(alpha=i)
    ridge =  linear_model.Ridge(alpha=i)
    # print(ms.cross_val_score(lasso, x, y, cv=5))



    # Mean_lasso_error = np.mean(ms.cross_val_score(lasso, x, y, cv=5,scoring='neg_mean_squared_error'))
    Mean_lasso_error = np.mean(ms.cross_val_score(lasso, x, y, cv=5,scoring='r2'))
    mean_lasso_err_list.append(Mean_lasso_error)
    # np.append(mean_lasso_err_list,Mean_lasso_error)
    if max_lasso_err<Mean_lasso_error:
        max_lasso_err = Mean_lasso_error
        max_lasso_alpha=i


    # Mean_ridge_error = np.mean(ms.cross_val_score(ridge, x, y, cv=5,scoring='neg_mean_squared_error'))
        Mean_ridge_error = np.mean(ms.cross_val_score(ridge, x, y, cv=5,scoring='r2'))

    mean_ridge_err_list.append(Mean_ridge_error)
    # mean_ridge_err_list.add(Mean_ridge_error)
    # np.append(mean_ridge_err_list,Mean_ridge_error)
    if max_ridge_err<Mean_ridge_error:
        max_ridge_err = Mean_ridge_error
        max_ridge_alpha=i


    print('When alpha = ',i,'Mean_lasso_error = ',Mean_lasso_error)
    print('When alpha = ',i,'Mean_ridge_error = ',Mean_ridge_error)
    print('max lasso alpha = ',max_lasso_alpha)
    print('max ridge alpha = ',max_ridge_alpha)
plt.scatter(alphalist,mean_lasso_err_list)
plt.scatter(alphalist,mean_ridge_err_list)
plt.plot(alphalist,mean_lasso_err_list,label='lasso')
plt.plot(alphalist,mean_ridge_err_list,label='ridge')
plt.title(' r2_score regarding with alpha in lasso and ridge')
plt.legend()
plt.show()
