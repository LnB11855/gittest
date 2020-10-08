import numpy as np
from numpy import genfromtxt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import seaborn as sns
from sklearn.model_selection import train_test_split

class codingAssignment(object):
    def __init__(self):
        self.data= {}
        self.model=None
    def data_read(self,csv_name=None):
        if not csv_name:
            X, y = make_classification(n_samples=500, n_features=20, n_redundant=2,n_repeated=2, n_informative=6,
                                       n_clusters_per_class=1, random_state=14)
            X=pd.DataFrame(X)
            for index in range(len(X.columns.values)):
                X.columns = X.columns.astype(str)
                X.columns.values[index]="x"+str(index)
            y = pd.DataFrame(y)
            y.columns=['y']
            self.data["features"]=X
            self.data["labels"]=y
        else:
            my_data = pd.read_csv(csv_name, delimiter=',')
            feature_col=[col for col in my_data if col.startswith('x')]
            self.data["features"]=my_data[feature_col]
            self.data["labels"]=my_data['y']
        print(self.data["features"].shape)
        print(self.data["labels"].shape)

    # def data_clean(self):
    # outliarn check, multicolinerity,precison,recall,f1,auc,roc,confusionmatrix,github

    def get_mode(self,model_name=None):
    #ensemble, unsupervised learning,
        if not model_name or model_name=="lr":
            self.model=LogisticRegression()
    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.data["features"], self.data["labels"].values.ravel(), test_size=0.2)
        self.model.fit(X_train,y_train)
        print(self.model.score(X_test,y_test))
    def indicator_convert(self,colname=None,thredshold_value=None):
        self.data["features"].loc[self.data["features"][colname]>thredshold_value],self.data["features"].loc[self.data["features"][colname]<=thredshold_value]=1,0
if __name__ == '__main__':

    ins = codingAssignment()
    ins.data_read()
    ins.get_mode()
    ins.train_model()
    ins.indicator_convert('x1',0)
    # print(ins.data["features"]['X1'])


