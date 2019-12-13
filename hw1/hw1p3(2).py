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



num = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]

num_lasso = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]

for i in range(1000):
    for j in range(13):
        num[j] += ridgeCoefficient[i][0][j]/1000
print(num)


for i in range(1000):
    for j in range(13):
        num_lasso[j] += lassoCoefficient[i][0][j]/1000
print(num_lasso)

#Lasso

b=[[1,2,3], [5,8], [7,8,9]]
num_lasso1=list(chain(*num_lasso))
plt.bar(range(len(num_lasso1)), num_lasso1)
plt.show()

#Ridge

b=[[1,2,3], [5,8], [7,8,9]]
num_ridge=list(chain(*num))
plt.bar(range(len(num_ridge)), num_ridge)
plt.show()
