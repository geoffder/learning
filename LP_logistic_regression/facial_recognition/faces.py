import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import shuffle

df = pd.read_csv('fer2013.csv')
# print(df.columns.values) # ['emotion' 'pixels' 'Usage']
# emotions = []
# for ele in df['emotion'].values:
#     if not ele in emotions:
#         emotions.append(ele)
# print(np.sort(emotions)) # [0 to 6]
# usages = []
# for ele in df['Usage'].values:
#     if not ele in usages:
#         usages.append(ele)
# print(usages)

# emotion: ints 0 through 6
# pixels: string object of ints seperated by spaces
# Usage: ['Training, 'PublicTest', 'PrivateTest']

# steps
# emotions are the classes, 7 of them. Only work on 0 and 1, so it is binary
# there are way more 0s than 1s though, need to build the dataset,
# so that n for each is equal can do this by repeating the data of the
# under-represented class (better than sub-sampling, more data) break the
# pixels string up, and make each pixel a feature. normalize the values
# don't really care about the Usage I don't think.
# I'll just shuffle to make my train and test sets

# how many of each class do I have in this set?
emotions = df['emotion'].values
T_0 = emotions[emotions == 0]  # create set for class 0
T_1 = emotions[emotions == 1]  # create set for class 1
# print('samples in class 0', T_0.shape[0])
# print('samples in class 1', T_1.shape[0])

pixels = df['pixels'].values
pixels_0 = pixels[emotions == 0]  # split pixels up in same way as emotions
pixels_1 = pixels[emotions == 1]
pix_0 = []
pix_1 = []
for i in range(len(pixels_0)):
    pixArray = pixels_0[i].split(' ')
    pix_0.append(list(map(int, pixArray)))
for i in range(len(pixels_1)):
    pixArray = pixels_1[i].split(' ')
    pix_1.append(list(map(int, pixArray)))

# make them numpy objects
X_0 = np.array(pix_0)
X_1 = np.array(pix_1)
# approximately equalize the sizes of each set
X_1 = np.concatenate(([pix_1]*(X_0.shape[0]//X_1.shape[0])), axis=0)
T_1 = np.concatenate(([T_1]*(len(T_0)//len(T_1))), axis=0)

# build X and Y, equalizing the representation of each class
X = np.concatenate((X_0, X_1), axis=0)
T = np.concatenate((T_0, T_1), axis=0)
N, D = X.shape
# normalize X values
X = (X - X.mean())/X.std()
# add bias term
ones = np.ones((N, 1))
Xb = np.concatenate((X, ones), axis=1)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def costDeriv(X, Y, T):
    return X.T @ (Y - T)
    # return X.T @ (T - Y) # trying adding weights up


def crossEntropy(Y, T):
    return -(T * np.log(Y) + (1 - T)*np.log(1 - Y)).mean()


def prediction_rate(P, T):
    return (T == P).mean()


# train and test sets
Xb, T = shuffle(Xb, T)
N_train = int(Xb.shape[0] * .8)
X_train = Xb[:N_train-1, :]
T_train = T[:N_train-1]
X_test = Xb[N_train:, :]
T_test = T[N_train:]

w = np.random.randn(D+1) / np.sqrt(D+1)
learning_rate = .000001
l2 = 2  # l2 regularlization constant

train_costs = []
test_costs = []
for i in range(3000):
    Y_train = sigmoid(X_train @ w)
    Y_test = sigmoid(X_test @ w)
    train_costs.append(crossEntropy(Y_train, T_train))
    test_costs.append(crossEntropy(Y_test, T_test))
    # w += learning_rate * (costDeriv(X_train, Y_train, T_train) - l2 * w)
    w -= learning_rate * (costDeriv(X_train, Y_train, T_train) + l2 * w)

# final probabilities, cost and prediction
Y_train = sigmoid(X_train @ w)
Y_test = sigmoid(X_test @ w)
train_costs.append(crossEntropy(Y_train, T_train))
test_costs.append(crossEntropy(Y_test, T_test))
P_train = np.round(Y_train)
P_test = np.round(Y_test)

print('training prediction rate:', prediction_rate(P_train, T_train))
print('test prediction rate:', prediction_rate(P_test, T_test))
plt.plot(train_costs, label='train costs')
plt.plot(test_costs, label='test costs')
plt.show()
