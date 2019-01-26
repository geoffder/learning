import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

import torch
from torch.autograd import Variable
from torch import optim

# use GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


class ANN(object):

    def __init__(self, hidden_layer_sizes, p_drop):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_rates = p_drop

    def fit(self, Xtrain, Ttrain, Xtest, Ttest, lr=1e-4, batch_mu=.1,
            epsilon=1e-4, epochs=40, batch_sz=200, print_every=50):

            N, D = Xtrain.shape
            K = np.unique(Ttrain).shape[0]

            Xtrain = torch.from_numpy(Xtrain).float().to(device)
            Ttrain = torch.from_numpy(Ttrain).long().to(device)
            Xtest = torch.from_numpy(Xtest).float().to(device)
            Ttest = torch.from_numpy(Ttest).long().to(device)

            self.model = torch.nn.Sequential()
            M1 = D  # first input layer is the number of features in X
            for i, M2 in enumerate(self.hidden_layer_sizes):
                self.model.add_module(
                    "dropout"+str(i+1),
                    torch.nn.Dropout(p=self.dropout_rates[i])
                )
                self.model.add_module(
                    "dense"+str(i+1),
                    torch.nn.Linear(M1, M2)
                )
                self.model.add_module(
                    "batchnorm"+str(i+1),
                    torch.nn.BatchNorm1d(
                        M2, eps=epsilon, momentum=batch_mu, affine=True,
                        track_running_stats=True
                    )
                )
                self.model.add_module(
                    "relu"+str(i+1),
                    torch.nn.ReLU()
                )
                M1 = M2  # input layer to next layer is this layer
            # output layer (no activation)
            self.model.add_module(
                "dropoutOut",
                torch.nn.Dropout(p=self.dropout_rates[-1])
            )
            self.model.add_module(
                "denseOut",
                torch.nn.Linear(M1, K)
            )

            self.model.to(device)

            self.loss = torch.nn.CrossEntropyLoss(size_average=True)
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

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

                    cost += self.train(Xbatch, Tbatch)  # train

                    if j % print_every == 0:
                        train_acc = self.score(Xtrain, Ttrain)
                        test_acc = self.score(Xtest, Ttest)
                        test_cost = self.get_cost(Xtest, Ttest)
                        print("cost: %f, acc: %.2f" % (test_cost, test_acc))

                # for plotting
                train_costs.append(cost / n_batches)
                train_accs.append(train_acc)
                test_costs.append(test_cost)
                test_accs.append(test_acc)

            plt.plot(train_costs, label='training cost')
            plt.plot(test_costs, label='validation cost')
            plt.legend()
            plt.show()
            plt.plot(train_accs, label='training accuracy')
            plt.plot(test_accs, label='validation accuracy')
            plt.legend()
            plt.show()

    def train(self, inputs, labels):
        # set the model to training mode
        # dropout and batch norm behave differently in train vs eval modes
        self.model.train()

        inputs = Variable(inputs, requires_grad=False)
        labels = Variable(labels, requires_grad=False)

        self.optimizer.zero_grad()  # Reset gradient

        # Forward
        logits = self.model.forward(inputs)
        output = self.loss.forward(logits, labels)

        # Backward
        output.backward()
        self.optimizer.step()  # Update parameters

        return output.item()

    # similar to train() but not doing the backprop step
    def get_cost(self, inputs, labels):
        # set the model to testing mode
        # dropout and batch norm behave differently in train vs eval modes
        self.model.eval()

        inputs = Variable(inputs, requires_grad=False)
        labels = Variable(labels, requires_grad=False)

        # Forward
        logits = self.model.forward(inputs)
        output = self.loss.forward(logits, labels)

        return output.item()

    # Note: inputs is a torch tensor
    def predict(self, inputs):
        # set the model to testing mode
        # dropout and batch norm behave differently in train vs eval modes
        self.model.eval()

        inputs = Variable(inputs, requires_grad=False)
        logits = self.model.forward(inputs)
        return logits.data.cpu().numpy().argmax(axis=1)

    def score(self, inputs, labels):
        predictions = self.predict(inputs)
        return np.mean(labels.cpu().numpy() == predictions)


def classRebalance(X, T):
    '''
    Take data and labels and increase number of samples for under-represented
    classes by duplicating the existing ones.
    '''
    classes = np.unique(T)
    Xlist = [X[T == k] for k in classes]
    Tlist = [T[T == k] for k in classes]
    bigN = np.max([t.shape for t in Tlist])
    # develop a less coarse way that better approximates the classes
    Xlist = [np.concatenate([x]*(bigN//x.shape[0]), axis=0) for x in Xlist]
    Tlist = [np.concatenate([t]*(bigN//t.shape[0]), axis=0) for t in Tlist]

    return np.concatenate(Xlist, axis=0), np.concatenate(Tlist, axis=0)


def trainTestSplit(X, T, ratio=.5):
    '''
    Shuffle dataset and split into training and validation sets given a
    train:test ratio.
    '''
    X, T = shuffle(X, T)
    N = X.shape[0]
    Xtrain, Ttrain = X[:int(N*ratio)], T[:int(N*ratio)]
    Xtest, Ttest = X[int(N*ratio):], T[int(N*ratio):]
    return Xtrain, Ttrain, Xtest, Ttest


def loadAndProcess(pca=True):
    print('loading in data...')
    if pca:
        X = pd.read_csv('pixels_300compPCA.csv').values
        T = pd.read_csv('emotion.csv').values.flatten()
    else:
        df = pd.read_csv('fer2013.csv')
        print('samples in full dataset:', df['emotion'].values.size)
        maxN = 20000
        print('taking %d samples' % maxN)
        X = np.array(
            [str.split(' ') for str in df['pixels'].values[:maxN]]
        ).astype(np.uint8)
        T = df['emotion'].values[:maxN]
    print('data loaded.')

    X, T = classRebalance(X, T)
    print('X shape:', X.shape, 'T shape:', T.shape)
    print('emotion counts:', [(T == k).sum() for k in np.unique(T)])
    return X, T


def main():
    X, T = loadAndProcess(pca=True)
    Xtrain, Ttrain, Xtest, Ttest = trainTestSplit(X, T, ratio=.8)

    ann = ANN([1000, 1000, 500, 500, 300, 100],
              [0.2, 0.5, 0.5, .5, .5, .5, .5])
    ann.fit(Xtrain, Ttrain, Xtest, Ttest, lr=1e-3, epochs=100)


if __name__ == '__main__':
    main()
