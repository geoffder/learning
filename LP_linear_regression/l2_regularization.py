import numpy as np
import matplotlib.pyplot as plt

# generate some data
N = 50
X = np.array(np.linspace(0, 10, N))
Y = np.array(.5*X + np.random.randn(N))
# manually make some outliers
Y[-1] += 30 # last two points higher than they should be
Y[-2] += 30

# # take a look at the data
# plt.scatter(X, Y)
# plt.show()

# practice
def simpleLinear(X, Y):
    # create an un-regularized fit
    denom = X.dot(X) - X.mean()*X.sum()
    a = (Y.dot(X) - X.mean() *Y.sum()) / denom
    b = (Y.mean()*X.dot(X) - X.mean()*Y.dot(X)) / denom
    Yhat = a*X + b

    plt.scatter(X,Y)
    plt.plot(X, Yhat)
    plt.show()
#simpleLinear(X,Y)

def regularized(X, Y, N):
    X = np.array([np.ones(N), X]).T # make 2 column matrix (bias term for w)
    # he does this: X = np.vstack([np.ones(N), X]).T (his X wasn't np.array yet though)

    # calculate weights of non-regularized solution (ml)
    w_ml = np.linalg.solve(X.T @ X, X.T @ Y) # maximum likelihood solution
    Yhat_ml = X @ w_ml

    # calculate L2 regularized solution
    l2 = 1000.0 # penalty strength, this is the lambda constant
    w_map = np.linalg.solve(l2 * np.eye(2) + X.T @ X, X.T @ Y) # eye(2) = 2x2 identity matrix
    Yhat_map = X @ w_map

    plt.scatter(X[:,1], Y)
    plt.plot(X[:,1], Yhat_ml, label='maximum likelihood')
    plt.plot(X[:,1], Yhat_map, label='maxima a posteriori')
    plt.legend()
    plt.show()

regularized(X, Y, N)
