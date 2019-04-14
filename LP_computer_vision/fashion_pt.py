import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

import torch
from torch import nn
from torch import optim

# use GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


class Flatten(nn.Module):
    'Layer that flattens extra-dimensional input to create an NxD matrix'
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class CNN(nn.Module):

    def __init__(self, dims, K, conv_layer_shapes, pool_szs,
                 hidden_layer_sizes, p_drop, batch_mu=.1, epsilon=1e-4):
        super(CNN, self).__init__()
        # data shape
        self.dims = dims  # input dimensions
        self.K = K  # output classes
        # architecture
        self.conv_layer_shapes = conv_layer_shapes
        self.pool_szs = pool_szs
        self.hidden_layer_sizes = hidden_layer_sizes
        self.drop_rates = p_drop
        # batch-norm hyper-parameters
        self.epsilon = epsilon
        self.batch_mu = batch_mu
        # assemble network and move to GPU
        self.build()
        self.to(device)

    def build(self):
        # convolutional layers
        self.conv_drops = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.maxpools = nn.ModuleList()
        self.conv_bnorms = nn.ModuleList()
        # fully connected layers
        self.dense_drops = nn.ModuleList()
        self.denses = nn.ModuleList()
        self.dense_bnorms = nn.ModuleList()

        for i, shape in enumerate(self.conv_layer_shapes):
            # dropout preceding covolutions
            self.conv_drops.append(nn.Dropout2d(p=self.drop_rates[i]))
            # convolutional layers (in features, out features, (H dim, W dim))
            self.convs.append(
                nn.Conv2d(shape[2], shape[3], (shape[0], shape[1]), stride=1,
                          padding=(shape[0]//2, shape[1]//2))
            )
            # stride for pool is same as kernel size by default
            self.maxpools.append(nn.MaxPool2d(self.pool_szs[i]))
            # batch normalization (pass through before non-linearity)
            self.conv_bnorms.append(nn.BatchNorm2d(shape[3]))

        # transform featuremaps into 1D vectors for transition to dense layers
        self.flatten = Flatten()

        # calculate dimensions going in to dense layers (post flattening len)
        num_fmaps = self.conv_layer_shapes[-1][3]
        pool_redux = np.prod([p for p in self.pool_szs])
        D = (
            np.floor(self.dims[0]/pool_redux)
            * np.floor(self.dims[1]/pool_redux)
            * num_fmaps
        ).astype(np.int)

        M1 = D  # input size to first dense layer (flattened conv output)
        for i, M2 in enumerate(self.hidden_layer_sizes):
            # dropout preceding fully-connected layer
            self.dense_drops.append(
                nn.Dropout(p=self.drop_rates[len(self.conv_drops)+i]))
            # fully-connected dense layer (linear transform)
            self.denses.append(nn.Linear(M1, M2))
            # batch-normalization preceding non-linearity
            self.dense_bnorms.append(
                nn.BatchNorm1d(M2, eps=self.epsilon, momentum=self.batch_mu,
                               affine=True, track_running_stats=True)
            )
            M1 = M2

        # output layers (final dropout and logistic regression)
        self.log_drop = nn.Dropout(p=self.drop_rates[-1])
        self.logistic = torch.nn.Linear(M1, self.K)

    def forward(self, X):
        # convolutional layers
        for drop, conv, pool, bnorm in zip(
                self.conv_drops, self.convs, self.maxpools, self.conv_bnorms):
            X = pool(nn.functional.elu(bnorm(conv(drop(X)))))
        X = self.flatten(X)
        # fully connected layers
        for drop, dense, bnorm in zip(
                self.dense_drops, self.denses, self.dense_bnorms):
            X = nn.functional.elu(bnorm(dense(X)))
        # get logits
        return self.logistic(self.log_drop(X))

    def fit(self, Xtrain, Ttrain, Xtest, Ttest, lr=1e-4, epochs=40,
            batch_sz=200, print_every=50):

            N = Xtrain.shape[0]  # number of samples

            # send data to GPU
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
                # shuffle dataset for next epoch of batches
                inds = torch.randperm(Xtrain.size()[0])
                Xtrain, Ttrain = Xtrain[inds], Ttrain[inds]
                for j in range(n_batches):
                    Xbatch = Xtrain[j*batch_sz:(j*batch_sz+batch_sz)]
                    Tbatch = Ttrain[j*batch_sz:(j*batch_sz+batch_sz)]

                    cost += self.train_step(Xbatch, Tbatch)

                    if j % print_every == 0:
                        # shuffle test set
                        inds = torch.randperm(Xtest.size()[0])
                        Xtest, Ttest = Xtest[inds], Ttest[inds]
                        # accuracies for train and test sets
                        train_acc = self.score(Xtrain, Ttrain)
                        test_cost, test_acc = self.cost_and_score(
                            Xtest, Ttest)
                        print("cost: %f, acc: %.2f" % (test_cost, test_acc))

                # for plotting
                train_costs.append(cost / n_batches)
                train_accs.append(train_acc)
                test_costs.append(test_cost)
                test_accs.append(test_acc)

            # plot cost and accuracy progression
            fig, axes = plt.subplots(1, 2)
            axes[0].plot(train_costs, label='training')
            axes[0].plot(test_costs, label='validation')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Cost')
            axes[1].plot(train_accs, label='training')
            axes[1].plot(test_accs, label='validation')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            fig.legend()
            fig.tight_layout()
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

        return output.item()

    def get_cost(self, inputs, labels):
        self.eval()  # set the model to testing mode
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

    def cost_and_score(self, inputs, labels):
        self.eval()  # set the model to testing mode
        with torch.no_grad():
            # Forward
            logits = self.forward(inputs)
            output = self.loss.forward(logits, labels)
        predictions = logits.data.cpu().numpy().argmax(axis=1)
        acc = np.mean(labels.cpu().numpy() == predictions)
        return output.item(), acc


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


def loadAndProcess():
    print('loading in data...')

    data = pd.read_csv('../large_files/Fashion_MNIST/fashion-mnist_test.csv')
    data = data.values  # convert to numpy
    print('samples in dataset:', data.shape[0])

    X = data[:, 1:].reshape(data.shape[0], 1, 28, 28)  # N x H x W x C
    T = data[:, 0]  # labels are first column
    print('X shape:', X.shape)
    print('T shape:', T.shape)

    return X, T


def conv_setup_1():
    convnet = CNN(
        [28, 28], 10,  # input dimesions and number of output classes
        [[5, 5, 1, 20], [5, 5, 20, 50], [5, 5, 50, 50]],  # conv layers
        [2, 2, 2],  # max pool sizes/strides
        [1000, 1000, 500, 500, 300, 100],  # fully connected layers
        [.2, .5, .5, .5, .5, .5, .5, .5, .5, .5],  # dropout rates
    )
    return convnet


def conv_setup_2():
    convnet = CNN(
        [28, 28], 10,  # input dimesions and number of output classes
        [[5, 5, 1, 20], [5, 5, 20, 40], [5, 5, 40, 60]],  # conv layers
        [2, 2, 2],  # max pool sizes/strides
        [1000, 500, 250, 100],  # fully connected layers
        [.2, .5, .5, .5, .5, .5, .5, .5],  # dropout rates
    )
    return convnet


def main():
    X, T = loadAndProcess()
    Xtrain, Ttrain, Xtest, Ttest = trainTestSplit(X, T, ratio=.8)
    del X, T  # free up memory
    convnet = conv_setup_1()
    convnet.fit(Xtrain, Ttrain, Xtest, Ttest, lr=1e-3, epochs=30)


if __name__ == '__main__':
    main()
