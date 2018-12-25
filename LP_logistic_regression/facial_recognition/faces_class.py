import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # don't need but maybe use to get familiar with API

from sklearn.utils import shuffle

def process(file):
    df = pd.read_csv(file) # 'fer2013.csv'

    # how many of each class do I have in this set?
    emotions = df['emotion'].values
    T_0 = emotions[emotions == 0] # create set for class 0
    T_1 = emotions[emotions == 1] # create set for class 1
    # print('samples in class 0', T_0.shape[0])
    # print('samples in class 1', T_1.shape[0])

    pixels = df['pixels'].values
    pixels_0 = pixels[emotions == 0] # split pixels up in same way as emotions
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
    ones = np.ones((N,1))
    Xb = np.concatenate((X, ones), axis=1)

    return Xb, T

class LogisticModel(object):

    def fit(self, X, T, learning_rate = 1e-6, l2reg = 2, epochs = 3000, show_fig=False): # he did 120000, no wonder I wasn't flat. lol

        # train and test sets
        N, D = X.shape
        X, T = shuffle(X, T)
        N_train = int(N * .8)
        X_train = X[:N_train-1,:]
        T_train = T[:N_train-1]
        X_test = X[N_train:,:]
        T_test = T[N_train:]

        self.w = np.random.randn(D) / np.sqrt(D)

        train_costs = []
        test_costs = []
        for i in range(epochs):
            Y_train = self.sigmoid(X_train @ self.w)
            Y_test = self.sigmoid(X_test @ self.w)
            train_costs.append(self.crossEntropy(Y_train, T_train))
            test_costs.append(self.crossEntropy(Y_test, T_test))
            #w += learning_rate * (costDeriv(X_train, Y_train, T_train) - l2reg * w)
            self.w -= learning_rate * (self.costDeriv(X_train, Y_train, T_train) + l2reg * self.w)

        # final probabilities, cost and prediction
        Y_train = self.sigmoid(X_train @ self.w)
        Y_test = self.sigmoid(X_test @ self.w)
        train_costs.append(self.crossEntropy(Y_train, T_train))
        test_costs.append(self.crossEntropy(Y_test, T_test))

        P_train = np.round(Y_train)
        P_test = np.round(Y_test)

        print('training prediction rate:', self.prediction_rate(P_train, T_train))
        print('test prediction rate:', self.prediction_rate(P_test, T_test))

        if show_fig:
            plt.plot(train_costs, label='train costs')
            plt.plot(test_costs, label='test costs')
            plt.show()

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def costDeriv(self, X, Y, T):
        return X.T @ (Y - T)
        #return X.T @ (T - Y) # trying adding weights up

    def crossEntropy(self, Y, T):
        return -(T * np.log(Y) + (1 - T)*np.log(1 - Y)).mean()

    def prediction_rate(self, P, T):
        return (T == P).mean()

def main():
    X, T = process('fer2013.csv')

    model = LogisticModel()
    model.fit(X, T, epochs=3000, show_fig=True)

if __name__ == '__main__':
    main()
