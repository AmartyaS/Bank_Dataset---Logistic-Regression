# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 09:45:50 2021

@author: amart
"""
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import pandas as pd

data=pd.read_csv("F:\\Data Science Assignments\\Python Codes\\bank.csv")
data.head()
data.columns

data['default']=data['default'].map({'yes':1,'no':0})
data['housing']=data['housing'].map({'yes':1,'no':0})
data['loan']=data['loan'].map({'yes':1,'no':0})
data['Target']=data['Target'].map({'yes':1,'no':0})

data.Target.value_counts()
data.loan.value_counts()
data.default.value_counts()
pd.crosstab(data.housing,data.Target)

yesTarget=len(data[data.Target==1])
noTarget=len(data[data.Target==0])
print("Percentage of people subscribed to term deposit : {:.2f}% ".format((yesTarget/len(data.Target))*100))
print("Percentage of people not-subscribed to term deposit :{:.2f}% ".format((noTarget/len(data.Target))*100))

noloan=len(data[data.loan==0])
yesloan=len(data[data.loan==1])
print("Percentage of people having loan : {:.2f}%".format((yesloan/len(data.loan))*100))
print("Percentage of people having no loan : {:.2f}%".format((noloan/len(data.loan))*100))

sb.countplot(x="loan",data=data,palette='bwr')
sb.countplot(x="Target",data=data,palette='bwr')

pd.crosstab(data.age,data.Target).plot(kind="bar",figsize=(20,6))
plt.title("Term Deposit Frequency based on Ages")
plt.xlabel("Age")
plt.ylabel("Frequency")

pd.crosstab(data.default,data.Target).plot(kind="bar",figsize=(10,6))
plt.title("Term Deposit Frequency based on Credit default")
plt.xlabel("Default")
plt.ylabel("Frequency")

pd.crosstab(data.housing,data.Target).plot(kind="bar")
plt.title("Term Deposit Frequency based on Housing Loan")
plt.xlabel("Housing")
plt.ylabel("Frequency")

pd.crosstab(data.loan,data.Target).plot(kind="bar")
plt.title("Term Deposit Frequency based on Loan")
plt.xlabel("Loan")
plt.ylabel("Frequency")

data.groupby('Target').mean()
data=data.drop(columns=["job","marital","education","contact","month","poutcome"])

data.corr()['Target'][:].plot.bar()

X=data.drop('Target',axis=1)
Y=data.Target
classifier=LogisticRegression()
classifier.fit(X,Y)

y_pred=classifier.predict(X)
confusion=confusion_matrix(Y, y_pred)
confusion
print(classification_report(Y, y_pred))

pd.crosstab(Y,y_pred)
fpr,tpr,thresholds=roc_curve(Y,classifier.predict_proba(X)[:,1])
auc=roc_auc_score(Y, y_pred)

plt.plot(fpr,tpr,color='orange',label='ROC Curve (area= %0.2f)'%auc)
plt.plot([0,1],[0,1],color='navy',linestyle='--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc='lower right')
