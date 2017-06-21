import os
import pandas as pd
import numpy as np
import time

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

train,test=sklearn.model_selection.train_test_split(fraud, test_size=.2, stratify=fraud['class'])

cols_to_use=['purchase_value', 'source', 'browser', 'sex', 'age', 'time_to_purchase', 'country', 'device_usage']

train_X=train[cols_to_use]

for i in cols_to_use:
    if fraud[i].dtype=='object':
        fraud[i+'_codes']=pd.Categorical(fraud[i]).codes
        cols_to_use[i].replace(i+'_codes')


train_Y=train['class']

