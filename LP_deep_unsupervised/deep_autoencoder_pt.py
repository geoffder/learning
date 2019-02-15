# import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
from torch.autograd import Variable
from torch import optim
from collections import OrderedDict

from LP_util import getKaggleMNIST

'''
This is my first try at this, and it works ok, but not as well as it could.
I need to try an architecture like LP uses, where the transposed weights of the
encoder layers are used for the output, rather than a seperate set of weights.
'''

# use GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


class DeepAutoEncoder(object):

    def __init__(self, nodes):
        self.nodes = nodes

    def build(self, D):
        self.model = torch.nn.Sequential(OrderedDict({}))

        M1 = D
        self.model._modules['dropout'] = torch.nn.Dropout(p=.5)
        for i, M2 in enumerate(self.nodes):
            self.model._modules[
                'hidden'+str(i)] = torch.nn.Linear(M1, M2)
            self.model._modules[
                'hidden_sigmoid'+str(i)] = torch.nn.Sigmoid()
            M1 = M2
        for i, M2 in enumerate(reversed([D]+self.nodes[:-1])):
            self.model._modules[
                'out'+str(len(self.nodes)-i)] = torch.nn.Linear(M1, M2)
            self.model._modules[
                'out_sigmoid'+str(i)] = torch.nn.Sigmoid()
            M1 = M2

        self.model.to(device)

    def fit(self, X, lr=1e-4, epochs=40, batch_sz=200, print_every=50):

        N, D = X.shape
        X = torch.from_numpy(X).float().to(device)

        self.build(D)  # build network (self.model)

        self.loss = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        n_batches = N // batch_sz
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
            '''
            items() gives (key, value) tuples of the dict, which can be indexed
            if turned in to a list. Then, take a slice that includes only the
            encoder (dense layers until centre, no drop) to do forward pass.
            '''
            for layer in list(
                    self.model._modules.items())[1:len(self.nodes)*2]:
                inputs = layer[1](inputs)
            reduced = inputs.cpu().numpy()
        return reduced


def main():
    Xtrain, Ttrain, _, _ = getKaggleMNIST()

    # hidden_layer_sizes = [1000, 800, 500, 300, 100, 10, 2]
    hidden_layer_sizes = [500, 300, 100, 10, 2]
    # hidden_layer_sizes = [1000, 500, 300, 100, 10, 3]
    # hidden_layer_sizes = [500, 300, 3]
    DAE = DeepAutoEncoder(hidden_layer_sizes)
    DAE.fit(Xtrain, lr=1e-4, epochs=10)

    reduced = DAE.get_reduced(Xtrain)
    print('reduced data shape:', reduced.shape)

    if hidden_layer_sizes[-1] == 2:
        fig, ax = plt.subplots(1, 1)
        ax.scatter(reduced[:, 0], reduced[:, 1], c=Ttrain,
                   cmap='jet', alpha=.5, s=80)
        ax.set_xlabel('component 1')
        ax.set_ylabel('component 2')
        plt.show()
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(
            reduced[:, 0], reduced[:, 1], reduced[:, 2],
            c=Ttrain, cmap='jet', alpha=.5, s=100
        )
        plt.show()


if __name__ == '__main__':
    main()
