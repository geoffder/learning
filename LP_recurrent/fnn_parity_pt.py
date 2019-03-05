import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from LP_util import all_parity_pairs
import torch
from torch import nn
from torch import optim

'''
Parity problem with a feed-forward neural net. There are 4096 unique sequence
labels for 12 bit sequences, which is likely why the effective layer sizes are
[2048] and [1024, 1024].
'''


class FNN(nn.Module):

    def __init__(self, nodes, D, K):
        super(FNN, self).__init__()
        self.nodes = nodes
        self.D = D
        self.K = K
        self.build()

    def build(self):
        M1 = self.D
        # lists of layers
        self.linears = nn.ModuleList()
        self.bnorms = nn.ModuleList()
        # main network initialization
        for i, M2 in enumerate(self.nodes):
            self.linears.append(nn.Linear(M1, M2))
            self.bnorms.append(nn.BatchNorm1d(M2))
            M1 = M2
        # logistic regression layer (no activation)
        self.logistic = nn.Linear(M1, self.K)
        # send to GPU if using it
        self.to(device)

    def forward(self, X):
        for linear, bnorm in zip(self.linears, self.bnorms):
            X = linear(X)
            X = bnorm(X)
            X = nn.functional.relu(X)
        return self.logistic(X)

    def fit(self, Xtrain, Ttrain, Xtest, Ttest, lr=1e-4,
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


def trainTestSplit(X, T, ratio=.8):
    '''
    Shuffle dataset and split into training and validation sets given a
    train:test ratio.
    '''
    X, T = shuffle(X, T)
    N = X.shape[0]
    Xtrain, Ttrain = X[:int(N*ratio)], T[:int(N*ratio)]
    Xtest, Ttest = X[int(N*ratio):], T[int(N*ratio):]
    return Xtrain, Ttrain, Xtest, Ttest


def main():
    nbits = 12
    X, T = all_parity_pairs(nbits)
    print('X shape:', X.shape, 'T shape:', T.shape)
    Xtrain, Ttrain, Xtest, Ttest = trainTestSplit(X, T)

    K = np.unique(T).shape[0]
    # hidden_layer_sizes = [2048]
    hidden_layer_sizes = [2048//2]*2
    # hidden_layer_sizes = [2048//4]*4
    # hidden_layer_sizes = [2048//8]*8
    # hidden_layer_sizes = [2048//16]*16

    ANN = FNN(hidden_layer_sizes, nbits, K)
    ANN.fit(Xtrain, Ttrain, Xtest, Ttest, lr=1e-2, epochs=50, batch_sz=200)


if __name__ == '__main__':
    # use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    main()
