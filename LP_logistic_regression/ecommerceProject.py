import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# note: this is how I wrote it without seeing him do it, I am leaving it as is
# since it still works. Can serve as a lesson and reminder of some functions
# and how things work, particularly pandas
def get_data():
    df = pd.read_csv('ecommerce_data.csv')
    print(df.head())

    # only want to work with user_actions 0 and 1

    # could drop all rows where users took other actions...
    for i in range(len(df['user_action'])):
        rows = []
        if df['user_action'][i] > 1:
            rows.append(i)
        df.drop(rows, inplace = True)
    # without reset_idex, row indices will be missing where they were removed
    df.reset_index(drop = True, inplace = True) # re-number rows to be continuous again
    #print(df.head())

    # or could change actions like begin and end purchase to 1, since the customer
    # would have to do that first to move to complete a purchase
    # df['user_action'] = np.ceil(df['user_action'] / 3.)

    # normalize non-binary data (n_products_viewed and visit_duration)
    df['n_products_viewed'] = (df['n_products_viewed'] - df['n_products_viewed'].mean())/ df['n_products_viewed'].std()
    df['visit_duration'] = (df['visit_duration'] - df['visit_duration'].mean())/ df['visit_duration'].std()

    N = len(df['user_action'])
    hot_time = np.array([[0]*df['time_of_day'][i]+[1]+[0]*(3 - df['time_of_day'][i]) for i in range(N)])
    df['12am_6am'] = hot_time[:,0]
    df['6am_12pm'] = hot_time[:,1]
    df['12pm_6pm'] = hot_time[:,2]
    df['6pm_12am'] = hot_time[:,3]
    df['ones'] = 1
    # print(df.head())

    # maybe take the ones off? not sure if done the same way for logistic
    #X = df[['is_mobile', 'n_products_viewed', 'visit_duration','is_returning_visitor', '12am_6am', '6am_12pm', '12pm_6pm', '6pm_12am', 'ones']]
    X = df[['is_mobile', 'n_products_viewed', 'visit_duration','is_returning_visitor', '12am_6am', '6am_12pm', '12pm_6pm', '6pm_12am']]
    X = X.values
    Y = df['user_action'].values

    return X, Y

X, Y = get_data()
N, D = X.shape
w = np.random.randn(D) / np.sqrt(D)
b = 0

z = X @ w + b

def sigmoid(z):
    return 1. / (1. + np.exp(-z))

Yhat = np.round(sigmoid(z)) # P_Y_given_X rounded to give 0s and 1s
mse = (Y - Yhat).T @ (Y - Yhat) / N

correct = 0
for i in range(len(Yhat)):
    if (Yhat[i] == Y[i]):
        correct += 1

print("percent correct = " + str(correct/N))
print("mse =", mse)
