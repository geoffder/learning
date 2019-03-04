import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim

'''
Experimental Layer to get more practice with making custom nn.Modules. The idea
is to allow strongly responding neurons to suppress others within the same
layer, like lateral inhibition. Consider layering then with static masking
so each set of weights (for each inhib layer) corresponds to a consistent
cluster of lateral neuron connections. NOTE: Screwing around here, likely to
not be a great idea.
'''


class Inhibitor(nn.Module):

    def __init__(self, M, mode='no_drop'):
        super(Inhibitor, self).__init__()
        self.M = M
        self.mode = mode
        self.W = nn.Parameter(torch.randn(M, M))
        self.bnorm = nn.BatchNorm1d(M)
        if mode == 'static_drop':
            self.mask = nn.Parameter(
                torch.bernoulli(torch.ones(M, M)*.2),
                requires_grad=False)

    def no_drop(self, X):
        # inhib = torch.sigmoid(X @ self.W)
        inhib = torch.sigmoid(self.bnorm(X @ self.W))
        return X - inhib*X

    def static_drop(self, X):
        maskW = self.mask @ self.W
        inhib = torch.sigmoid(X @ maskW)
        return X - inhib*X

    def dynamic_drop(self, X):
        # construct on GPU (no transfer)
        if self.training:
            mask = nn.Parameter(
                torch.bernoulli(
                    torch.cuda.FloatTensor(self.M, self.M).fill_(.5)),
                requires_grad=False)
            maskW = mask @ self.W
        else:
            maskW = self.W * 2
        inhib = torch.sigmoid(X @ maskW)
        return X - inhib*X

    def forward(self, X):
        if self.mode == 'no_drop':
            return self.no_drop(X)
        elif self.mode == 'static_drop':
            return self.static_drop(X)
        else:
            return self.dynamic_drop(X)


class InhibNeuralNet(nn.Module):

    def __init__(self, nodes, D, K):
        super(InhibNeuralNet, self).__init__()
        self.nodes = nodes
        self.D = D
        self.K = K
        self.build()

    def build(self):
        M1 = self.D
        # lists of layers
        self.linears = nn.ModuleList()
        self.inhibitors = nn.ModuleList()
        self.bnorms = nn.ModuleList()
        # main network initialization
        self.inDropout = nn.Dropout(p=.2)
        self.hiddenDropout = nn.Dropout(p=.5)
        for i, M2 in enumerate(self.nodes):
            self.linears.append(nn.Linear(M1, M2))
            self.bnorms.append(nn.BatchNorm1d(M2))
            self.inhibitors.append(Inhibitor(M2, mode='no_drop'))
            M1 = M2
        # logistic regression layer (no activation)
        self.logistic = nn.Linear(M1, self.K)
        # send to GPU if using it
        self.to(device)

    def forward(self, X):
        X = self.inDropout(X)
        for linear, bnorm, inhib in zip(
                self.linears, self.bnorms, self.inhibitors):
            X = linear(X)
            X = bnorm(X)
            X = inhib(X)
            X = nn.functional.relu(X)
            # X = self.hiddenDropout(X)
        return self.logistic(X)

    def fit(self, Xtrain, Ttrain, Xtest, Ttest, lr=1e-4, reg=1e-3,
            batch_mu=.1, epsilon=1e-4, mu=0.99, decay=0.99999,
            epochs=40, batch_sz=200, print_every=50):

            N = Xtrain.shape[0]

            Xtrain = torch.from_numpy(Xtrain).float().to(device)
            Ttrain = torch.from_numpy(Ttrain).long().to(device)
            Xtest = torch.from_numpy(Xtest).float().to(device)
            Ttest = torch.from_numpy(Ttest).long().to(device)

            self.loss = torch.nn.CrossEntropyLoss(reduction='mean').to(device)
            self.optimizer = optim.Adam(self.parameters(), lr=lr)

            n_batches = N // batch_sz
            train_costs, train_accs, test_costs, test_accs = [], [], [], []
            for i in range(epochs):
                cost = 0
                print("epoch:", i, "n_batches:", n_batches)
                inds = torch.randperm(Xtrain.size()[0])
                Xtrain, Ttrain = Xtrain[inds], Ttrain[inds]
                for j in range(n_batches):
                    Xbatch = Xtrain[j*batch_sz:(j*batch_sz+batch_sz)]
                    Tbatch = Ttrain[j*batch_sz:(j*batch_sz+batch_sz)]

                    cost += self.train_step(Xbatch, Tbatch)

                    if j % print_every == 0:
                        # inds = torch.randperm(Xtest.size()[0])
                        # Xtest, Ttest = Xtest[inds], Ttest[inds]
                        # train_acc = self.score(Xtrain[:1000], Ttrain[:1000])
                        # test_acc = self.score(Xtest[:1000], Ttest[:1000])
                        # test_cost = self.get_cost(Xtest[:1000], Ttest[:1000])
                        train_acc = self.score(Xtrain, Ttrain)
                        test_acc = self.score(Xtest, Ttest)
                        test_cost = self.get_cost(Xtest, Ttest)
                        print("cost: %f, acc: %.3f" % (test_cost, test_acc))

                        # for plotting
                        train_accs.append(train_acc)
                        test_accs.append(test_acc)
                test_costs.append(test_cost)
                train_costs.append(cost / n_batches)

            fig, axes = plt.subplots(1, 2)
            axes[0].plot(train_costs, label='training cost')
            axes[0].plot(test_costs, label='validation cost')
            axes[0].legend()
            axes[1].plot(train_accs, label='training accuracy')
            axes[1].plot(test_accs, label='validation accuracy')
            axes[1].legend()
            plt.show()

    def train_step(self, inputs, labels):
        # set the model to training mode
        # dropout and batch norm behave differently in train vs eval modes
        self.train()
        self.optimizer.zero_grad()  # Reset gradient

        # Forward
        logits = self.forward(inputs)
        output = self.loss.forward(logits, labels)

        # Backward
        output.backward()
        self.optimizer.step()  # Update parameters

        return output.item()

    def get_cost(self, inputs, labels):
        # set the model to testing mode
        # dropout and batch norm behave differently in train vs eval modes
        self.eval()

        'Wrap in no_grad to prevent graph from storing info for backprop'
        with torch.no_grad():
            # Forward
            logits = self.forward(inputs)
            output = self.loss.forward(logits, labels)

        return output.item()

    def predict(self, inputs):
        self.eval()
        with torch.no_grad():
            logits = self.forward(inputs)
        return logits.data.cpu().numpy().argmax(axis=1)

    def score(self, inputs, labels):
        predictions = self.predict(inputs)
        return np.mean(labels.cpu().numpy() == predictions)


def getKaggleMNIST():
    # MNIST data:
    # column 0 is labels
    # column 1-785 is data, with values 0 .. 255
    # total size of CSV: (42000, 1, 28, 28)
    train = pd.read_csv(
        '../large_files/digit_train.csv').values.astype(np.float32)
    train = shuffle(train)

    Xtrain = train[:-1000, 1:] / 255
    Ytrain = train[:-1000, 0].astype(np.int32)

    Xtest = train[-1000:, 1:] / 255
    Ytest = train[-1000:, 0].astype(np.int32)
    return Xtrain, Ytrain, Xtest, Ytest


def main(load=False):
    Xtrain, Ttrain, Xtest, Ttest = getKaggleMNIST()
    D = Xtrain.shape[1]
    K = np.unique(Ttrain).shape[0]

    hidden_layer_sizes = [1000, 500, 250, 10]

    INN = InhibNeuralNet(hidden_layer_sizes, D, K)
    INN.fit(Xtrain, Ttrain, Xtest, Ttest, lr=1e-2, epochs=20)


if __name__ == '__main__':
    # use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    main()
