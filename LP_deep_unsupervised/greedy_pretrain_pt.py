import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
from torch import optim
from collections import OrderedDict

from LP_util import getKaggleMNIST

# use GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


class AutoEncoder(object):

    def __init__(self, nodes):
        self.nodes = nodes

    def fit(self, X, lr=1e-4, epochs=40, batch_sz=200, print_every=50):

        N, D = X.shape
        X = torch.from_numpy(X).float().to(device)

        self.model = torch.nn.Sequential(OrderedDict({
            "dropout": torch.nn.Dropout(p=.5),
            "dense_hidden": torch.nn.Linear(D, self.nodes),
            "sigmoid1": torch.nn.Sigmoid(),
            "dense_out": torch.nn.Linear(self.nodes, D),
            "sigmoid2": torch.nn.Sigmoid(),
        }))
        self.model.to(device)

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

        return self.get_outputs(X)

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

    def get_outputs(self, inputs):
        'Return encoding weights and hidden representation of X'
        with torch.no_grad():
            inputs = Variable(inputs, requires_grad=False)

            activation = self.model._modules[
                "dense_hidden"](inputs).cpu().numpy()
            weights = self.model._modules["dense_hidden"].weight.cpu().numpy()

        return activation, weights


class ANN(object):

    def __init__(self, hidden_layer_sizes, weights):
        self.nodes = hidden_layer_sizes
        self.pre_weights = weights

    def fit(self, Xtrain, Ttrain, Xtest, Ttest, lr=1e-4, epochs=40,
            batch_sz=200, print_every=50):

        N, D = Xtrain.shape

        Xtrain = torch.from_numpy(Xtrain).float().to(device)
        Ttrain = torch.from_numpy(Ttrain).long().to(device)
        Xtest = torch.from_numpy(Xtest).float().to(device)
        Ttest = torch.from_numpy(Ttest).long().to(device)

        self.model = torch.nn.Sequential(OrderedDict({
            "dropout1": torch.nn.Dropout(p=.8),
            "dense1": torch.nn.Linear(D, self.nodes[0]),
            "relu1": torch.nn.ReLU(),
            "dropout2": torch.nn.Dropout(p=.5),
            "dense2": torch.nn.Linear(self.nodes[0], self.nodes[1]),
            "relu2": torch.nn.ReLU(),
            "dropout3": torch.nn.Dropout(p=.5),
            "dense3": torch.nn.Linear(self.nodes[1], self.nodes[2]),
            "relu3": torch.nn.ReLU(),
            "dense4": torch.nn.Linear(self.nodes[2], 10),  # logistic
        }))

        # load weights in manually to the dense layers
        self.model._modules["dense1"].weight.data = torch.Tensor(
            self.pre_weights[0])
        self.model._modules["dense2"].weight.data = torch.Tensor(
            self.pre_weights[1])
        self.model._modules["dense3"].weight.data = torch.Tensor(
            self.pre_weights[2])

        self.model.to(device)

        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
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

        fig, axes = plt.subplots(1, 2)
        axes[0].plot(train_costs, label='training cost')
        axes[0].plot(test_costs, label='validation cost')
        axes[0].legend()
        axes[1].plot(train_accs, label='training accuracy')
        axes[1].plot(test_accs, label='validation accuracy')
        axes[1].legend()
        fig.tight_layout()
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

        'Wrap in no_grad to prevent graph from storing info for backprop'
        with torch.no_grad():
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


def main():
    Xtrain, Ttrain, Xtest, Ttest, = getKaggleMNIST()

    hidden_layer_sizes = [500, 300, 100]

    ae_stack = [AutoEncoder(nodes) for nodes in hidden_layer_sizes]
    ae_weights = [[] for _ in range(len(hidden_layer_sizes))]
    input = Xtrain
    for i, ae in enumerate(ae_stack):
        print('#### BEGIN TRAINING FOR AUTOENCODER %s ####' % i)
        input, ae_weights[i] = ae.fit(input, lr=1e-3, epochs=20)

    # now train a network with these pre-trained weights
    ann = ANN(hidden_layer_sizes, ae_weights)
    print('#### BEGIN FINE TUNING AND CLASSIFICATION ####')
    ann.fit(Xtrain, Ttrain, Xtest, Ttest, lr=1e-3, epochs=30)


if __name__ == '__main__':
    main()
