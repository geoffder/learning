import numpy as np
import matplotlib.pyplot as plt
from LP_util import get_poetry_classifier_data
from sklearn.utils import shuffle

import torch
from torch import nn
from torch import optim


'''
Classification of sentences as being written by Robert Frost or Edgar Allan Poe
based on parts-of-speech tags (noun, pronoun, adverb, determiner, adjective,
etc). Last predicted label for each sequence is the decision for the sequence.
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

        N = len(X)

        # keep as lists (since each sample is a different size), but convert
        # all of the elements to tensors
        X = [torch.Tensor(ele).float().to(device) for ele in X]
        # extend label tensors for each sample to have proper sequence length
        Y = [torch.cuda.LongTensor(X[j].shape[0]).fill_(Y[j])
             for j in range(Y.shape[0])]
        # this also works
        # Y = [torch.Tensor([Y[j]]).long().expand(X[j].shape[0]).to(device)
        #      for j in range(Y.shape[0])]

        self.loss = torch.nn.CrossEntropyLoss(reduction='none').to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        epoch_costs, epoch_accs = [], []
        for i in range(epochs):
            epoch_cost, n_correct = 0, 0
            X, Y = shuffle(X, Y)
            for j in range(N):
                logits, cost = self.train_step(X[j], Y[j])
                # store costs
                epoch_cost += cost

                # check if final prediction is correct for current sample
                predictions = logits.data.cpu().numpy().argmax(axis=1)
                if predictions[-1] == Y[j][-1]:
                    n_correct += 1

            epoch_costs.append(epoch_cost)
            epoch_accs.append(n_correct/N)
            print("i:", i, "cost:", epoch_cost,
                  "classification rate:", n_correct/N)

        fig, axes = plt.subplots(1, 2)
        axes[0].plot(epoch_costs)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Cost')
        axes[1].plot(epoch_accs)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        fig.tight_layout()
        plt.show()

    def train_step(self, inputs, labels):
        self.train()  # set the model to training mode
        self.optimizer.zero_grad()  # Reset gradient

        # Forward
        logits = self.forward(inputs)
        output = self.loss.forward(logits, labels)

        # Backward
        # Using reduction='none' on CrossEntropyLoss since I only care about
        # the final label of the sequence. So, only take the last element of
        # the output Tensor. (.backward() only works on single element Tensors,
        # that's why reduction is normally done (sum or mean))
        output[-1].backward()
        # output.mean().backward()
        self.optimizer.step()  # Update parameters

        return logits, output[-1].item()


def one_hot_sequences(sequences, V):
    hot_mats = []
    for seq in sequences:
        matrix = np.zeros((len(seq), V))
        matrix[np.arange(len(seq)), seq] = 1
        hot_mats.append(matrix)
    return hot_mats


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


def main():
    X, Y, current_idx = get_poetry_classifier_data(1000)

    V = 0
    for seq in X:
        V = np.max(seq) if np.max(seq) > V else V
    V += 1
    print('Number of unique tags:', V)

    X = one_hot_sequences(X, V)

    K = 2
    nodes = [50]

    rnn = SimpleRNN(V, nodes, K)
    rnn.fit(X, Y, lr=1e-3, epochs=20)


if __name__ == '__main__':
    # use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    main()
