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