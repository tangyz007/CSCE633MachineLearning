import pandas as pd
import numpy as np
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from ipykernel import kernelapp as app
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, discriminant_analysis
from sklearn import datasets, linear_model, discriminant_analysis
from sklearn.linear_model import LogisticRegression
from itertools import chain

df = pd.read_csv('hw1_input.csv')
# df_replace=df.replace(['Yes','No'],['1','0'])
# df_replace=df_replace.replace(['Female','Male'],['1','0'])
# df_replace=df_replace.replace(['Asymptomatic','Abnormal','Angina','None'],['0','1','2','3'])
# df_replace=df_replace.replace(['norm',' hyper','abnorm'],['0','1','2'])
# df_replace=df_replace.replace(['Up','Flat','Down'],['0','1','2'])
# df_replace=df_replace.replace(['reversible Defect','Normal','Fixed Defect'],['0','1','2'])

df_replace = df.replace(['Male','Female','Yes','No','None','Angina','Abnormal','Asymptomatic',' hyper','abnorm','norm','Up','Flat','Down','reversible Defect','Fixed Defect','Normal'],
['1','0','1','0','3','2','1','0','2','1','0','2','1','0','2','1','0'])

x = df_replace[['Age','Sex','Chest Pain','BP','Cholestoral','fasting blood sugar > 120','resting ECG','max hr','angina','oldpeak','slope','major vessels','defect']]
x = np.array(x)
y = df_replace[['heart disease']]
y = np.array(y).reshape(-1)
#print(df.head(10))
bs_number = 1000
ridgeCoefficient = []
lassoCoefficient =[]
Scores = []
numberoftest  = int(0.2*len(df_replace))
numberoftrain = len(df_replace) - numberoftest


for _ in range(bs_number):
    train_set = np.asarray([True]*numberoftrain + [False]*numberoftest)
    np.random.shuffle(train_set)

    x_train, x_test, y_train, y_test = x[train_set], x[np.logical_not(train_set)], y[train_set], y[np.logical_not(train_set)]

    ridgeRegression = LogisticRegression(penalty = 'l2',solver='liblinear')
    ridgeRegression.fit(x_train, y_train)
    ridgeCoefficient.append(ridgeRegression.coef_)
    lassoRegression = LogisticRegression(penalty = 'l1',solver='liblinear')
    lassoRegression.fit(x_train, y_train)
    lassoCoefficient.append(lassoRegression.coef_)

# print(num_lasso)
# print('ridge mean coef',np.mean(ridgeCoefficient))
# print('lasso mean coef',np.mean(lassoCoefficient))
# print('ridge deviation coef',np.std(ridgeCoefficient))
# print('lasso deviation coef',np.std(lassoCoefficient))


num_ridge = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]
num_lasso = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]
#bootstraping 1000times
for i in range(1000):
    for j in range(13):
        num_ridge[j] += ridgeCoefficient[i][0][j]/1000
# print(num_ridge)


for i in range(1000):
    for j in range(13):
        num_lasso[j] += lassoCoefficient[i][0][j]/1000

import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc


y_test1 = list(map(int,y_test))

for _ in range(bs_number):
    train_set = np.asarray([True]*numberoftrain + [False]*numberoftest)
    np.random.shuffle(train_set)

    x_train, x_test, y_train, y_test = x[train_set], x[np.logical_not(train_set)], y[train_set], y[np.logical_not(train_set)]

    ridgeRegression = LogisticRegression(penalty = 'l2',solver='liblinear')
    ridgeRegression.fit(x_train, y_train)
    ridgeCoefficient.append(ridgeRegression.coef_)

    a = ridgeRegression.predict_proba(x_test)
    proba_ridge=a[:,1]
#     print(b)
    y_score_ridge = list(map(float,proba_ridge))
    fpr,tpr,threshold = roc_curve(y_test1, y_score_ridge)
    roc_auc = auc(fpr,tpr)

    lassoRegression = LogisticRegression(penalty = 'l1',solver='liblinear')
    lassoRegression.fit(x_train, y_train)
    lassoCoefficient.append(lassoRegression.coef_)
    ls=lassoRegression.predict_proba(x_test)
    proba_lasso=ls[:,1]
    y_score_lasso = list(map(float,proba_lasso))

    fpr1,tpr1,threshold1 = roc_curve(y_test1, y_score_lasso)
    roc_auc1 = auc(fpr1,tpr1)


    plt.figure()
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='red',lw=lw, label='ridge ROC curve (area = %0.2f)' % roc_auc)
    plt.plot(fpr1, tpr1, color='orange',lw=lw, label='ridge ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Ringe Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
