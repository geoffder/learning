import numpy as np
import pandas as pd

def get_data():
    df = pd.read_csv('ecommerce_data.csv')
    data = df.values

    X = data[:, :-1] # everything up to the last column
    Y = data[:, -1] # the last column

    # normalize non-binary data (n_products_viewed and visit_duration)
    X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
    X[:,2] = (X[:,2] - X[:,2].mean()) / X[:,2].std()

    # make a new, larger matrix to fill in the one hot encoded time features
    N, D = X.shape
    X2 = np.zeros((N, D+3))
    X2[:,0:(D-1)] = X[:,0:(D-1)] # copy in X

    # both of these options are probably more efficient and cleaner than
    # what I did in my blind attempt (ecommerceProject.py)
    for n in range(N):
        t = int(X[n, D-1]) # get the time of day value (0,1,2,3)
        X2[n, t+D-1] = 1 # put a 1 in the corresponding one-hot column (last 4)

    # or a fancier way
    Z = np.zeros((N, 4))
    # set indices corresponding to the value of time to 1
    Z[np.arange(N), X[:,D-1].astype(np.int32)] = 1

    return X2, Y

def get_binary_data():
    X, Y = get_data()
    X2 = X[Y <= 1] #only get Xs where Y is equal to 0 or 1
    # so my first guess that he would drop out all rows with actions besides 0 or 1 was correct
    Y2 = Y[Y <= 1] # remember this! never knew you could do this!
    return X2, Y2
