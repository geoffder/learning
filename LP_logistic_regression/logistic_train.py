import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from process import get_binary_data

X, Y = get_binary_data()
X, Y = shuffle(X, Y) # shuffle the order of samples
# then split the shuffled samples between train and test sets
Xtrain = X[:-100] # everything but the last 100 for train
Ytrain = Y[:-100]
Xtest = X[-100:] # the last 100 samples for test
Ytest = Y[-100:]

D = X.shape[1] # reminder: ones column was not added in process.py
W = np.random.randn(D) / np.sqrt(D)
b = 0

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward(X, W, b):
    return sigmoid(X @ W + b)

def classification_rate(Y, P):
    return np.mean(Y == P)

def cross_entropy(T, pY):
    #return -(T * np.log(Y) + (1 - T)*np.log(1 - Y)).sum() # it was just sum before..
    return -np.mean(T*np.log(pY) + (1 - T)*np.log(1 - pY)) # mean instead of sum?

train_costs = []
test_costs = []
learning_rate = .001

for i in range(1000):
    pYtrain = forward(Xtrain, W, b)
    pYtest = forward(Xtest, W, b)
    train_costs.append(cross_entropy(Ytrain, pYtrain))
    test_costs.append(cross_entropy(Ytest, pYtest))

    W -= learning_rate * (Xtrain.T @ (pYtrain - Ytrain))
    b -= learning_rate * (pYtrain - Ytrain).sum()

    if i % 1000 == 0:
        print(i, cross_entropy(Ytrain, pYtrain), cross_entropy(Ytest, pYtest))

print("Final train classification rate:", classification_rate(Ytrain, np.round(pYtrain)))
print("Final test classification rate:", classification_rate(Ytest, np.round(pYtest)))

legend1, = plt.plot(train_costs, label='train costs')
legend2, = plt.plot(test_costs, label='test costs')
plt.legend([legend1, legend2])
plt.show()
