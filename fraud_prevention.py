import os
import pandas as pd
import numpy as np
import sklearn
import sklearn.preprocessing
import sklearn.model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

os.chdir('/home/nick/metis_datasci')

fraud=pd.read_csv('Fraud_Data.csv')
country_ip=pd.read_csv('IpAddress_to_Country.csv')

fraud['purchase_time']=pd.to_datetime(fraud['purchase_time'])
fraud['signup_time']=pd.to_datetime(fraud['signup_time'])
fraud['time_to_purchase']=fraud['purchase_time']-fraud['signup_time']
fraud['time_to_purchase']=fraud['time_to_purchase'].dt.seconds

country_id=[]
for i in fraud['ip_address']:
    country_id.append(country_ip['country'][(i>=country_ip['lower_bound_ip_address']) \
                                            & (i<=country_ip['upper_bound_ip_address'])])

country_id_final=[]
for i in range(0, len(country_id)):
    if country_id[i].empty==False:
        country_id_final.append(country_id[i].iloc[0])
    else:
        country_id_final.append(np.nan)

fraud['country']=country_id_final

fraud['device_usage']=fraud['device_id'].map(fraud['device_id'].value_counts())
fraud['ip_usage']=fraud['ip_address'].map(fraud['ip_address'].value_counts())

cols_to_use=['purchase_value', 'source', 'browser', 'sex', 'age', 'time_to_purchase', 'country', 'device_usage', 'ip_usage']

for i in cols_to_use:
    if fraud[i].dtype=='object':
        fraud[i+'_codes']=pd.Categorical(fraud[i]).codes
        cols_to_use[cols_to_use.index(i)]=i+'_codes'

train,test=sklearn.model_selection.train_test_split(fraud, test_size=.2, stratify=fraud['class'])

train_X=train[cols_to_use]
train_Y=train['class']
test_X=test[cols_to_use]
test_Y=test['class']

model=LogisticRegression()
fit=model.fit(train_X, train_Y)
lreg_prediction=fit.predict(test_X)

pd.DataFrame(sklearn.metrics.confusion_matrix(test_Y, lreg_prediction), columns=['Pred -', 'Pred +'], \
             index=['Actual -', 'Actual +'])

print('Linear Regression Coefficients')
for i in range(0, len(test_X.keys())): print(fit.coef_[0][i], test_X.keys()[i])

model=RandomForestClassifier(150, oob_score=True, n_jobs=-1)
fit=model.fit(train_X, train_Y)
RF_prediction=fit.predict(test_X)

pd.DataFrame(sklearn.metrics.confusion_matrix(test_Y, RF_prediction), \
             columns=['Pred -', 'Pred +'], index=['Actual -', 'Actual +'])

print('Random Forest Feature Weights')
for i in range(0, len(test_X.keys())): print(fit.feature_importances_[i], test_X.keys()[i])