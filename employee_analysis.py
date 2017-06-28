import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn.apionly as sns
from datetime import timedelta

os.chdir('/home/nick/metis_datasci/')

employee_data=pd.read_csv('Employee_data.csv')

employee_data['join_date']=pd.to_datetime(employee_data['join_date'])
employee_data['quit_date']=pd.to_datetime(employee_data['quit_date'])
employee_data['tenure_length']=employee_data['quit_date']-employee_data['join_date']

company_headcount=pd.DataFrame()
dates=pd.date_range('2011/01/24', '2015/12/13')
dates_df=pd.DataFrame({'day': dates, 'key': np.full(len(dates),1)})

companies=sorted(employee_data['company_id'].unique())
companies_df=pd.DataFrame({'company_id': companies, 'key': np.full(len(companies), 1)})

headcount=pd.merge(dates_df, companies_df)
headcount.drop('key', axis=1, inplace=True)

num_joined=employee_data.groupby(['join_date', 'company_id']).apply(lambda x: len(x)).reset_index(name='num_joined')
num_quit=employee_data.groupby(['quit_date', 'company_id']).apply(lambda x: len(x)).reset_index(name='num_quit')

headcount = pd.merge(headcount, num_joined, how='left', left_on=['day', 'company_id'], right_on=['join_date', 'company_id'])
headcount = pd.merge(headcount, num_quit, how='left', left_on=['day', 'company_id'], right_on=['quit_date', 'company_id'])

headcount['num_joined']=headcount['num_joined'].fillna(0)
headcount['num_quit']=headcount['num_quit'].fillna(0)

headcount['join_cumsum']=headcount.groupby('company_id')['num_joined'].cumsum()
headcount['quit_cumsum']=headcount.groupby('company_id')['num_quit'].cumsum()

headcount['total_joined']=headcount['join_cumsum']-headcount['quit_cumsum']

for name, index in headcount.groupby('company_id'):
    plt.plot(dates, index['total_joined'])
plt.show()

plt.hist(employee_data['tenure_length'].dropna().dt.days, 100)
plt.show()

quitters=employee_data[employee_data['tenure_length'].dt.days<365]
lifers=employee_data[employee_data['tenure_length'].dt.days>365]

plt.boxplot((lifers['salary'].values, quitters['salary'].values))
plt.show()

sns.boxplot(data=(lifers['salary'].values, quitters['salary'].values))
plt.show()

sns.boxplot(x=employee_data['dept'], y=employee_data['salary'])
plt.show()

sns.boxplot(x=employee_data['company_id'], y=employee_data['tenure_length'])
plt.show()