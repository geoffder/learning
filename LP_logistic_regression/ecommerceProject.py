import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('ecommerce_data.csv')
print(df.head())

# only want to work with user_actions 0 and 1
for i in range(len(df['user_action'])):
    rows = []
    if df['user_action'][i] > 1:
        rows.append(i)
    df.drop(rows, inplace = True)
df.reset_index(drop = True, inplace = True) # re-number rows to be continuous again
print(df.head())

N = len(df['user_action'])
df['ones'] = 1
hot_time_of_day = [[0]*df['time_of_day'][i]+[1]+[0]*(3 - df['time_of_day'][i]) for i in range(N)]
# df['hot_time_of_day'] = [[0]*df['time_of_day'][i]+[1]+[0]*(3 - df['time_of_day'][i]) for i in range(N)]
# print(df.head())
X = df[['is_mobile', 'is_returning_visitor', 'ones']]
