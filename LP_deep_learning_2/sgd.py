import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from datetime import datetime

df = pd.read_csv('digit_train.csv')

labels = df['label'].values # .reshape(len(df['label']), 1)

X = df.loc[:, df.columns != 'label'].values
print('full X shape:', X.shape)

N, D = X.shape
K = len(set(labels))
T = np.zeros((N, K))
T[np.arange(N), labels.astype(np.int32)] = 1
print('full T shape:', T.shape)

#only taking a piece of the huge dataset
X, T = shuffle(X, T)
Xtr = X[:1000]
Ttr = T[:1000]
Xte = X[-1000:]
Tte = T[-1000:]

# use pca transformed dataset to keep dims down
Xtr = Xtr - Xtr.mean(axis=0) # centre dataset before PCA
Xte = Xte - Xte.mean(axis=0)

pca = PCA()
pca.fit_transform(Xtr) # fit and apply dimensionality transformation
pca.transform(Xte) # apply the same transformation to test set

# see plot_cumulative_variance in LP's utils.py for evidence of why 300 is reasonable
# (~95% + of variance is containted within the first 300)
D = 300 # number of columns we'll be taking from transformed data
Xtr = Xtr[:,:D]
Xte = Xte[:,:D]
Xtr = (Xtr - Xtr.mean(axis=0)) / (Xtr.std(axis=0) + 1e-5) # protect against div by zero
Xte = (Xte - Xte.mean(axis=0)) / (Xte.std(axis=0) + 1e-5)
# Xtr = (Xtr - Xtr.mean()) / Xtr.std()
# Xte = (Xte - Xte.mean()) / Xte.std()

# softmax
def softmax(A):
    return np.exp(A) / np.exp(A).sum(axis=1, keepdims=True)

# calculate Y
def forward(X, W, b):
    return softmax(X @ W + b)

# calculate derivative of objective/cost function
def costDeriv(X, Y, T):
    return X.T @ (Y - T)

def classification_rate(T, P):
    return np.mean(T == P)

def cost(T, Y): # multiclass
    return (T * np.log(Y)).sum()

# full gradient descent
learning_rate = .0001
l2 = 0.01 # L2 lambda
W = np.random.randn(D, K) / 28
b = np.zeros(K)

full_costs = []
t0 = datetime.now()
for i in range(200):
    Ytr = forward(Xtr, W, b)
    W -= learning_rate * (costDeriv(Xtr, Ytr, Ttr) + l2*W)
    b -= learning_rate * ((Ytr - Ttr).sum(axis=0)  + l2*b)

    Yte = forward(Xte, W, b)
    full_costs.append(cost(Tte, Yte))
    if(i % 20 == 0):
        print('iter ' + str(i) + ' test pred rate:', classification_rate(Tte, np.round(Yte)))

print('final test pred rate (full GD):', classification_rate(Tte, np.round(Yte)))
print('time taken (full GD):', datetime.now() - t0)

# stochastic gradient descent
learning_rate = .0001
l2 = 0.01 # L2 lambda
W = np.random.randn(D, K) / 28
b = np.zeros(K)

sto_costs = []
t0 = datetime.now()
for i in range(1):
    tmpX, tmpT = shuffle(Xtr, Ttr)
    for n in range(min(N, 500)): # not doing on whole set for time reasons
        xtr = tmpX[n, :].reshape(1,D)
        ttr = tmpT[n, :].reshape(1,K)
        ytr = forward(xtr, W, b)

        W -= learning_rate * (costDeriv(xtr, ytr, ttr) + l2*W)
        b -= learning_rate * ((ytr - ytr).sum(axis=0)  + l2*b)

        Yte = forward(Xte, W, b)
        sto_costs.append(cost(Tte, Yte))
        if(n % 50 == 0):
            print('iter ' + str(i) + ' test pred rate:', classification_rate(Tte, np.round(Yte)))

print('final test pred rate (stoch GD):', classification_rate(Tte, np.round(Yte)))
print('time taken (stoch GD):', datetime.now() - t0)

# batch gradient descent
learning_rate = .0001
l2 = 0.01 # L2 lambda
W = np.random.randn(D, K) / 28
b = np.zeros(K)

batch_sz = 200
n_batches = N // batch_sz

batch_costs = []
t0 = datetime.now()
for i in range(50):
    tmpX, tmpT = shuffle(Xtr, Ttr)
    for n in range(n_batches): # not doing on whole set for time reasons
        xtr = tmpX[n*batch_sz:n*batch_sz+batch_sz, :]
        ttr = tmpT[n*batch_sz:n*batch_sz+batch_sz, :]
        ytr = forward(xtr, W, b)

        W -= learning_rate * (costDeriv(xtr, ytr, ttr) + l2*W)
        b -= learning_rate * ((ytr - ttr).sum(axis=0)  + l2*b)

        Yte = forward(Xte, W, b)
        batch_costs.append(cost(Tte, Yte))
    if(i % 10 == 0):
        print('iter ' + str(i) + ' test pred rate:', classification_rate(Tte, np.round(Yte)))

print('final test pred rate (batch GD):', classification_rate(Tte, np.round(Yte)))
print('time taken (batch GD):', datetime.now() - t0)

x1 = np.linspace(0, 1, len(full_costs))
plt.plot(x1, full_costs, label='full GD')
x2 = np.linspace(0, 1, len(sto_costs))
plt.plot(x2, sto_costs, label='stochastic GD')
x3 = np.linspace(0, 1, len(batch_costs))
plt.plot(x3, batch_costs, label='batch GD')
plt.legend()
plt.show()

# note, batch is taking longer here because I am running on a very small set of the real dataset
# so the many batch iterations leads to more calculations, not fewer
