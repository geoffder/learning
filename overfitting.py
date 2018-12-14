import numpy as np
import matplotlib.pyplot as plt

# see bottom for execution block and global variables

# builds and returns a poly fitting X matrix
def make_poly(X, deg):
    n = len(X)
    data = [np.ones(n)] # create a column of ones (bias term)
    for d in range(deg):
        data.append(X**(d+1)) # thus cols = [ones, x, x^2, ..., x^deg]
    return np.vstack(data).T

def fit(X, Y):
    return np.linalg.solve(X.T @ X, X.T @ Y) # returns w

def fit_and_display(X, Y, sample, deg):
    N = len(X)
    train_idx = np.random.choice(N, sample) # take random samples from X and Y
    Xtrain = X[train_idx]
    Ytrain = Y[train_idx]

    # plt.scatter(Xtrain, Ytrain)
    # plt.show()

    # fit polynomial
    Xtrain_poly = make_poly(Xtrain, deg)
    w = fit(Xtrain_poly, Ytrain)

    # display the polynomial
    X_poly = make_poly(X, deg) # matrix to multiply against weights to make predictions
    Yhat = X_poly @ w
    # calculate r-squared
    SSres = ((Y - Yhat)**2).sum()
    SStot = ((Y - Y.mean())**2).sum()
    Rsq = 1. - SSres/SStot
    # plot
    plt.plot(X, Y)
    plt.plot(X, Yhat)
    plt.scatter(Xtrain, Ytrain)
    plt.title("deg = %d; R^2 = %f" % (deg, Rsq))
    plt.show()

# mean squared error
def get_mse(Y, Yhat):
    d = Y - Yhat
    return d.dot(d) / len(d)

def plot_train_vs_test_curves(X, Y, sample=20, max_deg=20):
    N = len(X)
    train_idx = np.random.choice(N, sample)
    Xtrain = X[train_idx]
    Ytrain = Y[train_idx]

    test_idx = [idx for idx in range(N) if idx not in train_idx]
    #test_idx = np.random.choice(N, sample)
    Xtest = X[test_idx]
    Ytest = Y[test_idx]

    mse_trains = []
    mse_tests = []

    for deg in range(max_deg+1):
        Xtrain_poly = make_poly(Xtrain, deg) # make X matrix for training data
        w = fit(Xtrain_poly, Ytrain) # calculate weights to fit training data
        Yhat_train = Xtrain_poly @ w # apply fit to training data

        Xtest_poly = make_poly(Xtest, deg) # make X matrix for test data
        Yhat_test = Xtest_poly @ w # apply fit to test data

        mse_trains.append(get_mse(Ytrain, Yhat_train))
        mse_tests.append(get_mse(Ytest, Yhat_test))

    plt.plot(range(max_deg+1), mse_trains, label="train mse")
    plt.plot(range(max_deg+1), mse_tests, label="test mse")
    plt.legend()
    plt.show()

# this block will only be executed if this "module" is run as a program
# i.e. from the command line with > python overfitting.py
# in this case, __name__ (controlling program/module) will be set to "__main__"
# if this module is instead imported by another module, then this block will not run
if __name__ == "__main__":
    # make up some data and take a look
    N = 100
    X = np.linspace(0, 6*np.pi, N)
    Y = np.sin(X)

    # plt.plot(X, Y)
    # plt.show()

    # run for increasing degrees
    # for deg in [2, 3, 4, 5, 6, 7 , 8 , 9]:
    #     fit_and_display(X, Y, 10, deg)

    plot_train_vs_test_curves(X, Y)
