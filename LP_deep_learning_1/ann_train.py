import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from process import get_data

X, T = get_data()

# get and set sizes
N, D = X.shape
M = 5 #3 I tried 3 at first, despite ann_predict using 5. Performs better with 5.
# He said it is complicated, picking the correct number of units, but it likely
# has some proportional relationship with how many features (D) and classes (K) there are
K = len(set(T))

# one-hot encoded targets
Thot = np.zeros((N, K))
Thot[np.arange(N), T.astype(np.int32)] = 1

# train and test set split
X, T, Thot = shuffle(X, T, Thot)

# I think in the future I'll just use the one-hot version
# then argmax both the T and Y matrices for prediction rate calculation
# lazy programmer does them seperately like this, for reference
Ntest = int(.2 * N) # 20% of the dataset will be used to test
Xtrain = X[:-Ntest,:]
Ttrain = T[:-Ntest]
THOTtrain = Thot[:-Ntest,:]
Xtest = X[-Ntest:,:]
Ttest = T[-Ntest:]
THOTtest = Thot[-Ntest:,:]

def softmax(A):
    return np.exp(A) / np.exp(A).sum(axis=1, keepdims=1)

def cost(T, Y):
    return -(T * np.log(Y)).mean()

def classification_rate(T, P):
    return (T == P).mean()

def deriv_w2(Z, T, Y):
    # (NxM).T @ (NxK) = (MxK)
    return Z.T @ (T - Y) # gradient ascent (T - Y)

def deriv_b2(T, Y):
    # (NxK).sum(axis=0) = (Kx1)
    return (T - Y).sum(axis=0)

def deriv_w1(X, Z, T, Y, W2):
    # (NxD).T @ ((NxK) @ (MxK).T * (NxM) * (1 - (NxM))
    # (NxD).T @ (NxM) = (DxM)
    return X.T @ ((T - Y) @ W2.T * Z * (1 - Z))

def deriv_b1(T, Y, W2, Z):
    # (NxK) @ (MxK).T * (NxM)
    # (NxM).sum(axis=0) = (Mx1)
    return ((T - Y) @ W2.T * Z * (1 - Z)).sum(axis=0)

def forward(X, W1, b1, W2, b2):
    # (NxD) @ (DxM) + (Mx1) = (NxM)
    Z = 1 / (1 + np.exp(-X @ W1 - b1))
    # (NxM) @ (MxK) + (Kx1) = (NxK)
    A = Z @ W2 + b2
    return softmax(A), Z

# initialize weights
W1 = np.random.randn(D,M)
b1 = np.random.randn(M)
W2 = np.random.randn(M,K)
b2 = np.random.randn(K)

learning_rate = .001
l2 = 0 #.0001 #screwing around with regularization
test_costs = []
train_costs = []
test_rates = []
train_rates = []
for epoch in range(10000):
    # calculate probabilites of Y given X
    Ytrain, Z = forward(Xtrain, W1, b1, W2, b2)
    Ytest, Ztest = forward(Xtest, W1, b1, W2, b2)
    # ascent gradients
    W2 += learning_rate * (deriv_w2(Z, THOTtrain, Ytrain) - l2 * W2)
    b2 += learning_rate * (deriv_b2(THOTtrain, Ytrain) - l2 * b2)
    W1 += learning_rate * (deriv_w1(Xtrain, Z, THOTtrain, Ytrain, W2) - l2 * W1)
    b1 += learning_rate * (deriv_b1(THOTtrain, Ytrain, W2, Z) - l2 * b1)

    P = np.argmax(Ytest, axis=1)
    if epoch % 100 == 0:
        # store costs
        train_c = cost(THOTtrain, Ytrain)
        test_c = cost(THOTtest, Ytest)
        Ptrain = np.argmax(Ytrain, axis=1)
        Ptest = np.argmax(Ytest, axis=1)
        train_r = classification_rate(Ttrain, Ptrain)
        test_r = classification_rate(Ttest, Ptest)
        print('train cost:', train_c, 'test cost:', test_c, 'train rate:', train_r, 'test rate:', test_r)
        train_costs.append(train_c)
        test_costs.append(test_c)
        train_rates.append(train_r)
        test_rates.append(test_r)

plt.plot(train_costs, label='train cost')
plt.plot(test_costs, label='test cost')
plt.legend()
plt.show()

plt.plot(train_rates, label='train rate')
plt.plot(test_rates, label='test rate')
plt.legend()
plt.show()
