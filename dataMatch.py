import pandas as pd
import seaborn as sns
import time
import matplotlib.pyplot as plt
import gc
import xgboost as xgb
from xgboost import plot_importance
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
coloumns=["Date","Quadrat","Blue","Green","Red","NIR","Rotation","NDVI","Inc"]
rawData=pd.read_csv("ALL_Bands_2016.csv")
rawData['Date'] = pd.to_datetime(rawData.Date)
cleanedDataOne=rawData[rawData['Date']==pd.datetime(2016,8,5)]
cleanedDataOne=cleanedDataOne.rename(columns={'Inc_Aug_5':'Inc'})
cleanedDataOne=cleanedDataOne[coloumns]

cleanedDataTwo=rawData[rawData['Date']==pd.datetime(2016,8,21)]
cleanedDataTwo=cleanedDataTwo.rename(columns={'Inc_Aug_22':'Inc'})
cleanedDataTwo=cleanedDataTwo[coloumns]

cleanedDataThree=rawData[rawData['Date']==pd.datetime(2016,8,31)]
cleanedDataThree=cleanedDataThree.rename(columns={'Inc_Aug_29':'Inc'})
cleanedDataThree=cleanedDataThree[coloumns]

cleanedData=pd.concat([cleanedDataThree])
# # fig, ax = plt.subplots(figsize=(10, 10))
# plot = sns.jointplot(x=cleanedData['SDS'], y=cleanedData['Inc_Sep_06'], kind='kde', color='blueviolet')
# plt.show()
df = cleanedData.groupby('Quadrat').cumcount()
print('grouping by Date,Quadrat, combines Blue_Mean')
gp =cleanedData[['Date','Quadrat','Blue']].groupby(['Date','Quadrat'])[['Blue']].mean().reset_index().rename(index=str, columns={'Blue': 'Blue_Mean'})
cleanedData = cleanedData.merge(gp, on=['Date','Quadrat'], how='left')
del gp
gc.collect()
print('grouping by Date,Quadrat, combines Blue_std')
gp =cleanedData[['Date','Quadrat','Blue']].groupby(['Date','Quadrat'])[['Blue']].std().reset_index().rename(index=str, columns={'Blue': 'Blue_std'})
cleanedData = cleanedData.merge(gp, on=['Date','Quadrat'], how='left')
del gp
gc.collect()
print('grouping by Date,Quadrat, combines Green_Mean')
gp =cleanedData[['Date','Quadrat','Green']].groupby(['Date','Quadrat'])[['Green']].mean().reset_index().rename(index=str, columns={'Green': 'Green_Mean'})
cleanedData = cleanedData.merge(gp, on=['Date','Quadrat'], how='left')
del gp
gc.collect()
print('grouping by Date,Quadrat, combines Green_std')
gp =cleanedData[['Date','Quadrat','Green']].groupby(['Date','Quadrat'])[['Green']].std().reset_index().rename(index=str, columns={'Green': 'Green_std'})
cleanedData = cleanedData.merge(gp, on=['Date','Quadrat'], how='left')
del gp
gc.collect()
print('grouping by Date,Quadrat, combines Red_Mean')
gp =cleanedData[['Date','Quadrat','Red']].groupby(['Date','Quadrat'])[['Red']].mean().reset_index().rename(index=str, columns={'Red': 'Red_Mean'})
cleanedData = cleanedData.merge(gp, on=['Date','Quadrat'], how='left')
del gp
gc.collect()
print('grouping by Date,Quadrat, combines Red_std')
gp =cleanedData[['Date','Quadrat','Red']].groupby(['Date','Quadrat'])[['Red']].std().reset_index().rename(index=str, columns={'Red': 'Red_std'})
cleanedData = cleanedData.merge(gp, on=['Date','Quadrat'], how='left')
del gp
gc.collect()
print('grouping by Date,Quadrat, combines NIR_Mean')
gp =cleanedData[['Date','Quadrat','NIR']].groupby(['Date','Quadrat'])[['NIR']].mean().reset_index().rename(index=str, columns={'NIR': 'NIR_Mean'})
cleanedData = cleanedData.merge(gp, on=['Date','Quadrat'], how='left')
del gp
gc.collect()
print('grouping by Date,Quadrat, combines NIR_std')
gp =cleanedData[['Date','Quadrat','NIR']].groupby(['Date','Quadrat'])[['NIR']].std().reset_index().rename(index=str, columns={'NIR': 'NIR_std'})
cleanedData = cleanedData.merge(gp, on=['Date','Quadrat'], how='left')
del gp
gc.collect()
print('grouping by Date,Quadrat, combines NDVI_Mean')
gp =cleanedData[['Date','Quadrat','NDVI']].groupby(['Date','Quadrat'])[['NDVI']].mean().reset_index().rename(index=str, columns={'NDVI': 'NDVI_Mean'})
cleanedData = cleanedData.merge(gp, on=['Date','Quadrat'], how='left')
del gp
gc.collect()
print('grouping by Date,Quadrat, combines NDVI_std')
gp =cleanedData[['Date','Quadrat','NDVI']].groupby(['Date','Quadrat'])[['NDVI']].std().reset_index().rename(index=str, columns={'NDVI': 'NDVI_std'})
cleanedData = cleanedData.merge(gp, on=['Date','Quadrat'], how='left')
del gp
gc.collect()
cleanedData['Date']=pd.to_datetime(cleanedData['Date'])
cleanedData.loc[cleanedData['Rotation']=="S4",'Rotation']=1
cleanedData.loc[cleanedData['Rotation']=="S3",'Rotation']=0
cleanedData.loc[cleanedData['Rotation']=="S2",'Rotation']=-1

# standardCol=['Blue_Mean', 'Blue_std', 'Green_Mean', 'Green_std', 'Red_Mean', 'Red_std', 'NIR_Mean', 'NIR_std', 'NDVI_Mean', 'NDVI_std']
# cleanedData[standardCol]=StandardScaler().fit_transform(cleanedData[standardCol].values)
#

trainData=cleanedData[cleanedData['Quadrat']<=4000]
testData=cleanedData[cleanedData['Quadrat']>4000]
dropCol=['Date','Inc','Blue','Green','Red','NIR','NDVI','Quadrat']
dropCol=['Date','Inc','Quadrat']
x1=trainData.drop(columns=dropCol)
y1=trainData['Inc']
x2=testData.drop(columns=dropCol)
y2=testData['Inc']
print(x1.shape)
print(x2.shape)


start_time = time.time()
threadHold=5
params = {'eta': 0.3,
          'tree_method': "hist",
          'grow_policy': "lossguide",
          'max_leaves': 1400,
          'max_depth': 0,
          'subsample': 0.9,
          'colsample_bytree': 0.7,
          'colsample_bylevel':0.7,
          'min_child_weight':0,
          'alpha':4,
          'objective': 'reg:linear',

          'eval_metric': 'rmse',
          'nthread':8,
          'random_state': 99,
          'silent': True}



# x1, x2, y1, y2 = train_test_split(Input, Output, test_size=0.1, random_state=99)

dtrain = xgb.DMatrix(x1, y1)
dvalid = xgb.DMatrix(x2, y2)

watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
model = xgb.train(params, dtrain, 200, watchlist, early_stopping_rounds=50, verbose_eval=5)
plot_importance(model)
#
pred= model.predict(xgb.DMatrix(x1), ntree_limit=model.best_ntree_limit)
pred=np.mat(pred).T
truth=y1.reset_index().drop(columns=['index']).values
pred[pred<threadHold]=0
pred[pred>threadHold]=1
truth[truth<threadHold]=0
truth[truth>threadHold]=1
Train_acc=np.sum(pred==truth)/pred.shape[0]

pred= model.predict(xgb.DMatrix(x2), ntree_limit=model.best_ntree_limit)
pred=np.mat(pred).T
truth=y2.reset_index().drop(columns=['index']).values
pred[pred<threadHold]=0
pred[pred>threadHold]=1
truth[truth<threadHold]=0
truth[truth>threadHold]=1
Test_acc=np.sum(pred==truth)/pred.shape[0]