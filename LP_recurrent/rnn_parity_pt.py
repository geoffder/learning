import numpy as np
import matplotlib.pyplot as plt

from LP_util import all_parity_pairs_with_sequence_labels
import torch
from torch import nn
from torch import optim


'''
Parity problem with a simple recurrent neural net. My own implementation
mirroring the way I built the tensorflow simple RNN. Here, no 'scan' function
is used due to pytorch's dynamic graph. I've just done it with a python loop,
whether there is a faster/better way to do this, I'll find out eventually.
'''


class RNNunit(nn.Module):
    def __init__(self, M1, M2):
        super(RNNunit, self).__init__()
        self.M1 = M1
        self.M2 = M2
        self.build()

    def build(self):
        self.Wx = nn.Parameter(torch.randn(self.M1, self.M2)
                               / np.sqrt(self.M1 + self.M2))  # input weight
        self.Wh = nn.Parameter(torch.randn(self.M2, self.M2)  # hidden weight
                               / np.sqrt(self.M1 + self.M2))
        self.bh = nn.Parameter(torch.zeros(self.M2))  # hidden bias
        self.h0 = nn.Parameter(torch.zeros(1, self.M2))  # initial hidden state

    def forward(self, X):
        X = X @ self.Wx  # multiply input by input weights
        # tensor to store intermediate outputs in during loop (and return)
        out = nn.Parameter(
                torch.cuda.FloatTensor(X.shape[0], self.M2).fill_(0),
                requires_grad=False)
        hidden = self.h0  # first pass uses initial values (0) for hidden
        for i in range(X.shape[0]):
            hidden = nn.functional.relu(X[i] + hidden @ self.Wh + self.bh)
            out[i, :] = hidden
        return out


class SimpleRNN(nn.Module):

    def __init__(self, D, nodes, K):
        super(SimpleRNN, self).__init__()
        self.D = D
        self.nodes = nodes
        self.K = K
        self.build()

    def build(self):
        M1 = self.D
        # lists of layers
        self.RNNunits = nn.ModuleList()
        # main network initialization
        for i, M2 in enumerate(self.nodes):
            self.RNNunits.append(RNNunit(M1, M2))
            M1 = M2
        # logistic regression layer (no activation)
        self.logistic = nn.Linear(M1, self.K)
        # send to GPU if using it
        self.to(device)

    def forward(self, X):
        for rnn in self.RNNunits:
            X = rnn.forward(X)
        return self.logistic(X)

    def fit(self, X, Y, lr=1e-2, epochs=40, batch_sz=200, print_every=50):

        N, T, D = X.shape

        X = torch.from_numpy(X).float().to(device)
        Y = torch.from_numpy(Y).long().to(device)

        self.loss = torch.nn.CrossEntropyLoss(reduction='mean').to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        sample_costs = []
        for i in range(epochs):
            epoch_cost, n_correct = 0, 0
            print("epoch:", i)
            inds = torch.randperm(X.size()[0])
            X, Y = X[inds], Y[inds]
            for j in range(N):
                logits, cost = self.train_step(X[j].reshape(T, D), Y[j])

                # store costs
                epoch_cost += cost
                sample_costs.append(cost)

                # check if final prediction is correct for current sample
                predictions = logits.data.cpu().numpy().argmax(axis=1)
                if predictions[-1] == Y[j, -1]:
                    n_correct += 1

            print("i:", i, "cost:", epoch_cost,
                  "classification rate:", (float(n_correct)/N))

            # stop if perfect accuracy is attained
            if n_correct == N:
                break

        plt.plot(sample_costs)
        plt.xlabel('Samples Ran')
        plt.ylabel('Sample Cost')
        plt.show()

    def train_step(self, inputs, labels):
        self.train()  # set the model to training mode
        self.optimizer.zero_grad()  # Reset gradient

        # Forward
        logits = self.forward(inputs)
        output = self.loss.forward(logits, labels)

        # Backward
        output.backward()
        self.optimizer.step()  # Update parameters

        return logits, output.item()


def main():
    nbits = 12
    X, Y = all_parity_pairs_with_sequence_labels(nbits)
    print('X shape:', X.shape, 'Y shape:', Y.shape)
    X = X.astype(np.float32)

    N, T, D = X.shape
    K = 2
    nodes = [20]

    rnn = SimpleRNN(D, nodes, K)
    rnn.fit(X, Y, lr=1e-2, epochs=200)


if __name__ == '__main__':
    # use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    main()
