import numpy as np
import pandas as pd

# columns of ecommerce data:
# is_mobile,n_products_viewed,visit_duration,is_returning_visitor,time_of_day,user_action
def get_data():

    df = pd.read_csv('ecommerce_data.csv')
    data = df.values

    X = data[:, :-1] # everything up to the last column
    Y = data[:, -1] # the last column
    N, D = X.shape

    # normalize non-integer values
    X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std() # n_products_viewed
    X[:,2] = (X[:,2] - X[:,2].mean()) / X[:,2].std() # visit_duration

    # one-hot encode time_of_day
    hot_day = np.zeros((N, np.int(X[:,D-1].max()+1)))
    hot_day[np.arange(N), X[:,D-1].astype(np.int32)] = 1

    X = np.concatenate((X[:,:-1], hot_day), axis=1) # exclude original time column

    return(X, Y)

#get_data()
