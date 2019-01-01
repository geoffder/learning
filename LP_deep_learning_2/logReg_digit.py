import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.utils import shuffle
from sklearn.decomposition import PCA

# load data
df = pd.read_csv('digit_train.csv')

# images are 28x28, 784 pixels.
#print(df.head())

labels = df['label'].values # .reshape(len(df['label']), 1)

# for i in set(labels):
#     temp = df.loc[df.label == i]
#     print('class '+str(i)+ ' samples: ', temp.shape[0])
# looks like there is not a class imbalance, fine to proceed

X = df.loc[:, df.columns != 'label'].values
X = (X - X.mean())/X.std() # shouldnt be doing like this
# Xmu = X.mean(axis=0)
# Xstd = X.std(axis=0)
# X = (X - Xmu)/(Xstd + 1e-11) # but the std on one of my features seems to be 0
#X = (X - X.mean(axis=0)) / X.std(axis=0) # so it is giving a div by 0 error
# what is best practice for this?
print('X shape:', X.shape)

N, D = X.shape
K = len(set(labels))
T = np.zeros((N, K))
T[np.arange(N), labels.astype(np.int32)] = 1
print('T shape:', T.shape)

#only taking a piece of the huge dataset
X, T = shuffle(X, T)
Xtr = X[:3000]
Ttr = T[:3000]
Xte = X[-3000:]
Tte = T[-3000:]

# try trimming down dimensions with PCA first
# makes the process faster as the low information features have been removed
if(0):
    pca = PCA(n_components=300)
    pca.fit(Xtr.T)
    Xtr = pca.components_.T
    pca.fit(Xte.T)
    Xte = pca.components_.T
    D = 300

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

learning_rate = .0001
l2 = 0 # L2 lambda
W = np.random.randn(D, K)
b = np.random.randn(K)

train_costs = []
test_costs = []
for i in range(200):
    Ytr = forward(Xtr, W, b)
    Yte = forward(Xte, W, b)
    W -= learning_rate * costDeriv(Xtr, Ytr, Ttr)
    b -= learning_rate * (Ytr - Ttr).sum(axis=0)
    train_costs.append(cost(Ttr, Ytr))
    test_costs.append(cost(Tte, Yte))
    if(i % 20 == 0):
        print('train pred rate:', classification_rate(Ttr, np.round(Ytr)))
        print('test pred rate:', classification_rate(Tte, np.round(Yte)))

print('train pred rate:', classification_rate(Ttr, np.round(Ytr)))
print('test pred rate:', classification_rate(Tte, np.round(Yte)))

plt.plot(train_costs)
plt.plot(test_costs)
plt.show()
