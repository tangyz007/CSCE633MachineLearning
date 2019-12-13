import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('hw1_input.csv', index_col=0, parse_dates=True)
# print(df.head(5))
# print(df.shape)
# df_all=df.replace(['Female','Male','Yes','No','None','Angina','Abnormal','Asymptomatic',' hyper','abnorm','norm','Up','Flat','Down','reversible Defect','Fixed Defect','Normal']
# ,['1','0','1','0','3','2','1','0','2','1','0','2','1','0','2','1','0'])


df_all=df.replace(['Yes','No'],['1','0'])
df_all=df_all.replace(['Female','Male'],['1','0'])
df_all=df_all.replace(['Asymptomatic','Abnormal','Angina','None'],['0','1','2','3'])
df_all=df_all.replace(['norm',' hyper','abnorm'],['0','1','2'])
df_all=df_all.replace(['Up','Flat','Down'],['0','1','2'])
df_all=df_all.replace(['reversible Defect','Normal','Fixed Defect'],['0','1','2'])

# print(df_all.head(20))
sns.pairplot(df,hue="heart disease")
plt.show()
