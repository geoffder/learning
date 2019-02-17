import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
from torch.autograd import Variable
from torch import optim
from collections import OrderedDict

from LP_util import getKaggleMNIST

'''
Have not figured out where I am going wrong with this yet. When using the
sigmoids, the values are saturating and it is just learning an average blob.
Even when using some batch normalization the problem persists. The encoder
network paired with decode function seems to work for a pure linear network,
but something is happening that is preventing learning in the non-linear case.
Try building one by creating a custom autoencoder module (inherit nn.Module).
'''

# use GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


class DeepAutoEncoder(object):

    def __init__(self, nodes):
        self.nodes = nodes

    def build(self):
        self.encoder = torch.nn.Sequential(OrderedDict({}))
        self.decoder_biases = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros(M))
             for M in reversed([self.D]+self.nodes[:-1])]
        )

        M1 = self.D
        # self.encoder._modules['dropout'] = torch.nn.Dropout(p=.5)
        for i, M2 in enumerate(self.nodes):
            self.encoder._modules[
                'hidden'+str(i)] = torch.nn.Linear(M1, M2)
            # self.encoder._modules[
            #     'hidden_bnorm'+str(i)] = torch.nn.BatchNorm1d(
            #         M2,  # eps=epsilon, momentum=batch_mu,
            #         affine=True, track_running_stats=True
            #     )
            self.encoder._modules[
                'hidden_sigmoid'+str(i)] = torch.nn.Sigmoid()
            M1 = M2

        self.encoder.to(device)
        self.decoder_biases.to(device)

    def fit(self, X, lr=1e-4, epochs=40, batch_sz=200, print_every=50):

        self.N, self.D = X.shape
        X = torch.from_numpy(X).float().to(device)

        self.build()  # build network

        # self.loss = torch.nn.MSELoss()
        self.loss = torch.nn.BCELoss()
        params = list(self.encoder.parameters()) + list(self.decoder_biases)
        self.optimizer = optim.Adam(params, lr=lr)
        # self.optimizer = torch.optim.RMSprop(
        #     params, lr=lr, alpha=0.99, eps=1e-08, weight_decay=0,
        #     momentum=0, centered=False)

        n_batches = self.N // batch_sz
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

    def decode(self, output):
        for i, bias in enumerate(self.decoder_biases):
            W = self.encoder._modules['hidden'+str(len(self.nodes)-i-1)].weight
            output = torch.nn.functional.linear(
                output,
                W.transpose(0, 1),
                bias=bias
            )
            output = output.sigmoid()
        return output

    def train(self, inputs):
        # set the model to training mode
        # dropout and batch norm behave differently in train vs eval modes
        self.encoder.train()

        inputs = Variable(inputs, requires_grad=False)

        self.optimizer.zero_grad()  # Reset gradient

        # Forward
        reduced = self.encoder.forward(inputs)
        reconstruction = self.decode(reduced)
        output = self.loss.forward(reconstruction, inputs)

        # Backward
        output.backward()
        self.optimizer.step()  # Update parameters

        return output.item()

    # similar to train() but not doing the backprop step
    def get_cost(self, inputs):
        # set the model to testing mode
        # dropout and batch norm behave differently in train vs eval modes
        self.encoder.eval()

        'Wrap in no_grad to prevent graph from storing info for backprop'
        with torch.no_grad():
            inputs = Variable(inputs, requires_grad=False)

            # Forward
            reduced = self.encoder.forward(inputs)
            reconstruction = self.decode(reduced)
            output = self.loss.forward(reconstruction, inputs)

        return reconstruction, output.item()

    def get_reduced(self, X):
        'Return reduced dimensionality hidden representation of X'
        X = torch.from_numpy(X).float().to(device)
        self.encoder.eval()
        with torch.no_grad():
            inputs = Variable(X, requires_grad=False)
            reduced = self.encoder.forward(inputs).cpu().numpy()
        return reduced

    def reconstruct(self, X):
        for i in range(10):
            idx = np.random.randint(0, X.shape[0], 1)
            sample = torch.from_numpy(X[idx]).float().to(device)

            self.encoder.eval()

            with torch.no_grad():
                inputs = Variable(sample, requires_grad=False)

                # self.optimizer.zero_grad()  # Reset gradient

                # Forward
                reduced = self.encoder.forward(inputs)
                reconstruction = self.decode(reduced).cpu().numpy()

            plt.imshow(reconstruction.reshape(28, 28), cmap='gray')
            plt.show()
            again = input("Show another reconstruction? Enter 'n' to quit\n")
            if again == 'n':
                break


def display_hidden():
    Xtrain, Ttrain, _, _ = getKaggleMNIST()

    hidden_layer_sizes = [500, 300, 100]
    DAE = DeepAutoEncoder(hidden_layer_sizes)
    DAE.fit(Xtrain, lr=1e-5, epochs=20)
    DAE.reconstruct(Xtrain)


def main():
    Xtrain, Ttrain, _, _ = getKaggleMNIST()
    # Xtrain = (Xtrain - Xtrain.mean())/(Xtrain.std() + .000001)

    # hidden_layer_sizes = [1000, 800, 500, 300, 100, 10, 2]
    # hidden_layer_sizes = [500, 300, 100, 10, 2]
    hidden_layer_sizes = [500, 300, 2]
    # hidden_layer_sizes = [1000, 500, 300, 100, 10, 3]
    # hidden_layer_sizes = [500, 300, 3]

    DAE = DeepAutoEncoder(hidden_layer_sizes)
    DAE.fit(Xtrain, lr=1e-5, epochs=20)

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
    # display_hidden()
