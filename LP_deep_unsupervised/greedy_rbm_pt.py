import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim

from LP_util import getKaggleMNIST


class RBM(nn.Module):
    'Restricted Boltzmann Machine Custom Module'
    def __init__(self, D, nodes):
        super(RBM, self).__init__()
        self.D = D
        self.M = nodes
        self.build()

    def build(self):
        self.W = nn.Parameter(
            torch.randn(self.D, self.M) * np.sqrt(2.0 / self.M))
        self.c = nn.Parameter(torch.zeros(self.M))
        self.b = nn.Parameter(torch.zeros(self.D))
        self.to(device)

    def sample_h(self, v):
        p_h_given_v = torch.sigmoid(v @ self.W + self.c)
        return torch.bernoulli(p_h_given_v)

    def sample_v(self, h):
        p_v_given_h = torch.sigmoid(h @ self.W.t() + self.b)
        return torch.bernoulli(p_v_given_h)

    def forward(self, V):
        return self.sample_v(self.sample_h(V))

    def forward_hidden(self, X):
        return torch.sigmoid(X @ self.W + self.c)

    def free_energy(self, V):
        first_term = V @ self.b.reshape(-1,)
        second_term = torch.sum(nn.functional.softplus(V @ self.W + self.c), 1)
        return first_term - second_term

    def fit(self, X, lr=1e-4, epochs=40, batch_sz=200, print_every=50,
            testing=False):
        N = X.shape[0]

        if not torch.is_tensor(X):
            X = torch.from_numpy(X).float().to(device)

        self.loss = FakeLoss()  # free energy v minus free energy v'
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        n_batches = N // batch_sz
        costs = []
        for i in range(epochs):
            cost = 0
            if testing:
                print("epoch:", i, "n_batches:", n_batches)
            inds = torch.randperm(X.size()[0])
            X = X[inds]
            for j in range(n_batches):
                Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz)]

                cost = self.train_step(Xbatch)  # train

                if j % print_every == 0 and testing:
                    print("cost: %f" % (cost))
            costs.append(cost)

        if testing:
            plt.plot(costs)
            plt.show()

    def train_step(self, V):
        self.train()  # put this module into training mode
        self.optimizer.zero_grad()  # Reset gradient

        # Forward
        V_prime = self.forward(V)
        output = self.loss.forward(
            self.free_energy(V), self.free_energy(V_prime))

        # Backward
        output.backward()
        self.optimizer.step()  # Update parameters

        return output.item()

    def transform(self, X):
        self.eval()
        with torch.no_grad():
            H = torch.sigmoid(X @ self.W + self.c)
        return H


class FakeLoss(torch.autograd.Function):
    '''
    Custom loss function for Restricted Boltzmann Machines. Forward is written,
    inputs are free energy of input (original visible layer) and the free
    energy of the probability of visible given hidden.
    '''
    @staticmethod
    def forward(fe_v, fe_vprime):  # , bias=None):
        output = torch.mean(fe_v - fe_vprime)
        return output


class PreTrainedANN(nn.Module):

    def __init__(self, D, nodes):
        super(PreTrainedANN, self).__init__()
        self.D = D
        self.nodes = nodes
        self.build()

    def build(self):
        'Create network with stack of Restricted Boltzmann Machines'
        self.RBMs = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.dropouts.append(nn.Dropout(p=.2))  # input layer dropout

        M1 = self.D
        for M2 in self.nodes:
            self.RBMs.append(RBM(M1, M2))
            self.dropouts.append(nn.Dropout(p=.5))
            M1 = M2

        self.denseOut = nn.Linear(M1, 10)
        self.to(device)

    def pretrain(self, X, pre_epochs):
        for rbm in self.RBMs:
            rbm.fit(X, lr=1e-4, epochs=pre_epochs)
            X = rbm.transform(X)

    def fit(self, Xtrain, Ttrain, Xtest, Ttest, lr=1e-4, pre_epochs=3,
            epochs=40, batch_sz=200, print_every=50):

        N = Xtrain.shape[0]

        Xtrain = torch.from_numpy(Xtrain).float().to(device)
        Ttrain = torch.from_numpy(Ttrain).long().to(device)
        Xtest = torch.from_numpy(Xtest).float().to(device)
        Ttest = torch.from_numpy(Ttest).long().to(device)

        if pre_epochs > 0:
            # pre-train stacked Restricted Boltzmann Machines
            print('#### BEGIN RBM PRE-TRAINING ####')
            self.pretrain(Xtrain, pre_epochs)

        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        print('#### BEGIN CLASSIFICATION TUNING ####')
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

                cost += self.train_step(Xbatch, Tbatch)  # train

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

        fig, axes = plt.subplots(1, 2)
        axes[0].plot(train_costs, label='training cost')
        axes[0].plot(test_costs, label='validation cost')
        axes[0].legend()
        axes[1].plot(train_accs, label='training accuracy')
        axes[1].plot(test_accs, label='validation accuracy')
        axes[1].legend()
        fig.tight_layout()
        plt.show()

    def train_step(self, inputs, labels):
        self.train()

        self.optimizer.zero_grad()  # Reset gradient

        # Forward
        logits = self.forward(inputs)
        output = self.loss.forward(logits, labels)

        # Backward
        output.backward()
        self.optimizer.step()  # Update parameters

        return output.item()

    # similar to train() but not doing the backprop step
    def get_cost(self, inputs, labels):
        self.eval()

        'Wrap in no_grad to prevent graph from storing info for backprop'
        with torch.no_grad():
            # Forward
            logits = self.forward(inputs)
            output = self.loss.forward(logits, labels)

        return output.item()

    def forward(self, X):
        X = self.dropouts[0](X)
        for rbm, drop in zip(self.RBMs, self.dropouts):
            X = drop(rbm.forward_hidden(X))
        return self.denseOut(X)

    def predict(self, inputs):
        self.eval()

        logits = self.forward(inputs)
        return logits.data.cpu().numpy().argmax(axis=1)

    def score(self, inputs, labels):
        predictions = self.predict(inputs)
        return np.mean(labels.cpu().numpy() == predictions)


def single_rbm():
    Xtrain, Ttrain, Xtest, Ttest, = getKaggleMNIST()

    rbm = RBM(Xtrain.shape[1], 100)
    rbm.fit(Xtrain, lr=1e-4, epochs=5, testing=True)


def main():
    Xtrain, Ttrain, Xtest, Ttest = getKaggleMNIST()

    hidden_layer_sizes = [500, 300, 100]
    ANN = PreTrainedANN(Xtrain.shape[1], hidden_layer_sizes)
    ANN.fit(Xtrain, Ttrain, Xtest, Ttest, lr=1e-4, pre_epochs=3, epochs=20)


if __name__ == '__main__':
    # use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    main()
    # single_rbm()
