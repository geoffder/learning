import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from mpl_toolkits.mplot3d import Axes3D

import torch
from torch.autograd import Variable
from torch import optim

from LP_util import getKaggleMNIST


'''
Deep AutoEncoder using custom class (inheriting nn.Module) in pytorch. Tied
weights are used, unlike deep_autoencoder_pt.py which learns separate weights
on the encoder and decoder sides.
'''


class DeepAutoEncoderModule(torch.nn.Module):

    def __init__(self, nodes, D):
        super(DeepAutoEncoderModule, self).__init__()
        self.nodes = nodes
        self.D = D
        self.build()

    def build(self):
        M1 = self.D
        'Encoder layers'
        self.dropout = torch.nn.Dropout(p=.5)
        self.weights = torch.nn.ParameterList()
        self.hidden_biases = torch.nn.ParameterList()
        self.hidden_bnorms = torch.nn.ModuleList()
        for i, M2 in enumerate(self.nodes):
            self.weights.append(torch.nn.Parameter(torch.randn(M1, M2)))
            self.hidden_biases.append(torch.nn.Parameter(torch.zeros(M2)))
            self.hidden_bnorms.append(torch.nn.BatchNorm1d(M2))
            M1 = M2
        'Decoder layers (weights are tied, so only biases)'
        self.out_biases = torch.nn.ParameterList()
        self.out_bnorms = torch.nn.ModuleList()
        for M in reversed([self.D] + self.nodes[:-1]):
            self.out_biases.append(torch.nn.Parameter(torch.zeros(M)))
            self.out_bnorms.append(torch.nn.BatchNorm1d(M))

    def encode(self, X):
        X = self.dropout(X)
        for W, b, bnorm in zip(
                self.weights, self.hidden_biases, self.hidden_bnorms):
            X = X @ W + b
            X = bnorm(X)
            X = torch.sigmoid(X)
        return X

    def decode(self, X):
        for W, b, bnorm in zip(
                reversed(self.weights), self.out_biases, self.out_bnorms):
            X = X @ W.transpose(0, 1) + b
            X = bnorm(X)
            X = torch.sigmoid(X)
        return X

    def forward(self, X):
        return self.decode(self.encode(X))


class DeepAutoEncoder(object):

    def __init__(self, nodes):
        self.nodes = nodes

    def fit(self, X, lr=1e-4, epochs=40, batch_sz=200, print_every=50):

        N, D = X.shape
        X = torch.from_numpy(X).float().to(device)

        self.model = DeepAutoEncoderModule(self.nodes, D)
        self.model.to(device)

        self.loss = torch.nn.MSELoss()
        # self.loss = torch.nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        n_batches = N // batch_sz
        costs = []
        for i in range(epochs):
            cost = 0
            print("epoch:", i, "n_batches:", n_batches)
            inds = torch.randperm(X.size()[0])
            X = X[inds]
            for j in range(n_batches):
                Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz)]

                cost = self.train(Xbatch)  # train

                if j % print_every == 0:
                    print("cost: %f" % (cost))
            costs.append(cost)

        plt.plot(costs)
        plt.show()

    def train(self, inputs):
        # set the model to training mode
        # dropout and batch norm behave differently in train vs eval modes
        self.model.train()

        inputs = Variable(inputs, requires_grad=False)

        self.optimizer.zero_grad()  # Reset gradient

        # Forward
        reconstruction = self.model.forward(inputs)
        output = self.loss.forward(reconstruction, inputs)

        # Backward
        output.backward()
        self.optimizer.step()  # Update parameters

        return output.item()

    # similar to train() but not doing the backprop step
    def get_cost(self, inputs):
        # set the model to testing mode
        # dropout and batch norm behave differently in train vs eval modes
        self.model.eval()

        'Wrap in no_grad to prevent graph from storing info for backprop'
        with torch.no_grad():
            inputs = Variable(inputs, requires_grad=False)

            # Forward
            reconstruction = self.model.forward(inputs)
            output = self.loss.forward(reconstruction, inputs)

        return reconstruction, output.item()

    def get_reduced(self, X):
        'Return reduced dimensionality hidden representation of X'
        X = torch.from_numpy(X).float().to(device)
        with torch.no_grad():
            inputs = Variable(X, requires_grad=False)
            reduced = self.model.encode(inputs).cpu().numpy()
        return reduced

    def reconstruct(self, X):
        for i in range(50):
            idx = np.random.randint(0, X.shape[0], 1)
            sample = torch.from_numpy(X[idx]).float().to(device)

            self.model.eval()

            with torch.no_grad():
                inputs = Variable(sample, requires_grad=False)

                # Forward
                reconstruction = self.model.forward(inputs).cpu().numpy()

            fig, axes = plt.subplots(1, 2)
            axes[0].imshow(X[idx].reshape(28, 28), cmap='gray')
            axes[1].imshow(reconstruction.reshape(28, 28), cmap='gray')
            plt.show()

            again = input(
                "Show another reconstruction? Enter 'n' to quit\n")
            if again == 'n':
                break


def display_hidden(load=False):
    Xtrain, Ttrain, _, _ = getKaggleMNIST()

    hidden_layer_sizes = [500, 300, 10]

    DAE = DeepAutoEncoder(hidden_layer_sizes)
    DAE.fit(Xtrain, lr=1e-2, epochs=10)

    DAE.reconstruct(Xtrain)


def main(load=False):
    Xtrain, Ttrain, Xtest, Ttest = getKaggleMNIST()

    # hidden_layer_sizes = [1000, 800, 500, 300, 100, 10, 2]
    # hidden_layer_sizes = [500, 300, 100, 10, 2]
    # hidden_layer_sizes = [500, 300, 2]
    # hidden_layer_sizes = [1000, 500, 300, 100, 10, 3]
    hidden_layer_sizes = [500, 300, 3]

    DAE = DeepAutoEncoder(hidden_layer_sizes)
    DAE.fit(Xtrain, lr=1e-2, epochs=10)

    reduTrain = DAE.get_reduced(Xtrain)
    reduTest = DAE.get_reduced(Xtest)
    print('reduced data shape:', reduTrain.shape)

    if hidden_layer_sizes[-1] == 2:
        fig, ax = plt.subplots(1, 2)
        for k in range(10):
            ax[0].scatter(reduTrain[Ttrain == k, 0], reduTrain[Ttrain == k, 1],
                          alpha=.5, s=80, label=k)
            ax[1].scatter(reduTest[Ttest == k, 0], reduTest[Ttest == k, 1],
                          alpha=.5, s=80)
        ax[0].set_title('training data')
        ax[0].set_xlabel('component 1')
        ax[0].set_ylabel('component 2')
        ax[1].set_title('test data')
        ax[1].set_xlabel('component 1')
        ax[1].set_ylabel('component 2')
        fig.legend()
        # fig.tight_layout()
        plt.show()
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for k in range(10):
            ax.scatter(reduTest[Ttest == k, 0], reduTest[Ttest == k, 1],
                       reduTest[Ttest == k, 2], alpha=.5, s=80, label=k)
        ax.legend()
        plt.show()


if __name__ == '__main__':
    # change colours used for sequential plotting
    new_colors = [plt.get_cmap('jet')(1. * i/10) for i in range(10)]
    plt.rc('axes', prop_cycle=(cycler('color', new_colors)))

    # use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # main()
    display_hidden()
