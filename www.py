import numpy as np
from numpy import genfromtxt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.linear_model import LogisticRegression,RidgeClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_classification
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score,precision_score,roc_auc_score
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
# outliarn check, multicolinerity,precison,recall,f1,auc,roc,confusionmatrix,github
my_data = pd.read_csv("/Users/biluning/Downloads/DataSet2.csv", delimiter=',')
feature_col = [col for col in my_data if col.startswith('x')]
x = my_data[feature_col]
y = my_data['y']
print(x.shape)
print(y.shape)
model=LogisticRegression()
model=SVC()
model=RidgeClassifier()
num_fold=5
sfolder = StratifiedKFold(n_splits=num_fold,random_state=64)
#---------neumeric analysis-------------
sns.countplot(x="x1",data=x,hue=y)
plt.show()
sns.countplot(y)
plt.show()
sns.boxplot(data=x)
plt.show()
# x=x.drop(columns=["x1"])



one_hot = pd.get_dummies(x['x1'])
# Drop column B as it is now encoded
x = x.drop('x1',axis = 1)
# Join the encoded df
# x = x.join(one_hot)
def checkVIF(df):
    X=add_constant(df)
    return pd.Series([variance_inflation_factor(X.values, i)
               for i in range(X.shape[1])],
              index=X.columns)
print(checkVIF(x))
#accuracy,f1,recall,precision,auc
history=defaultdict(list)

for train, test in sfolder.split(x,y):
    xtrain=x.iloc[train]
    xtest=x.iloc[test]
    ytrain=y.iloc[train].values.ravel()
    ytest=y.iloc[test].values.ravel()
    model.fit(xtrain.values,ytrain)
    #predict labels
    pred_train=model.predict(xtrain)
    pred_test=model.predict(xtest)

    #accuracy
    history["train_acc"].append(accuracy_score(ytrain,pred_train))
    history["test_acc"].append(accuracy_score(ytest,pred_test))
    #f1
    history["train_f1"].append(f1_score(ytrain,pred_train))
    history["test_f1"].append(f1_score(ytest,pred_test))
    #recall
    history["train_recall"].append(recall_score(ytrain,pred_train))
    history["test_recall"].append(recall_score(ytest,pred_test))
    #precision
    history["train_precision"].append(precision_score(ytrain,pred_train))
    history["test_precision"].append(precision_score(ytest,pred_test))
    # #predict probablities
    # prob_train=model.predict_proba(xtrain)
    # prob_test=model.predict_proba(xtest)
    # #aucroc
    # history["train_rocauc"].append(roc_auc_score(ytrain,prob_train[:,1]))
    # history["test_rocauc"].append(roc_auc_score(ytest,prob_test[:,1]))




print("Train acc  is %0.2f%% Test acc is %0.2f%%"%(np.mean(history["train_acc"])*100,np.mean(history["test_acc"])*100))
print("Train f1  is %0.2f%% Test f1 is %0.2f%%"%(np.mean(history["train_f1"])*100,np.mean(history["test_f1"])*100))
print("Train recall  is %0.2f%% Test recall is %0.2f%%"%(np.mean(history["train_recall"])*100,np.mean(history["test_recall"])*100))
print("Train precision  is %0.2f%% Test precision is %0.2f%%"%(np.mean(history["train_precision"])*100,np.mean(history["test_precision"])*100))
print("Train rocauc  is %0.2f%% Test rocauc is %0.2f%%"%(np.mean(history["train_rocauc"])*100,np.mean(history["test_rocauc"])*100))
thredshold_value=0
# data["features"].loc[data["features"][colname]>thredshold_value],data["features"].loc[data["features"][colname]<=thredshold_value]=1,0


