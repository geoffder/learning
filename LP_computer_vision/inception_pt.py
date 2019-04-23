import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

# use GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


class Flatten(nn.Module):
    "Layer that flattens extra-dimensional input to create an NxD matrix"
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class InceptionBlock(nn.Module):
    """
    Implementation of an inception style convolutional block. For simplicity,
    this block will follow a stereotyped structure, but the number of
    features maps at each stage are specified as follows:
        [in features, bottleneck features, out features]
    Note that since each of the 4 branches within the block are concatenated,
    the number of features maps of the output of this block is the specified
    'out features' * 4.
    """
    def __init__(self, features):
        super(InceptionBlock, self).__init__()
        self.features = features  # [in, bottleneck, out]
        self.build()

    def build(self):
        # branch 1: (1x1) conv only
        self.br1_conv1 = nn.Conv2d(
                self.features[0], self.features[2], (1, 1), stride=1,
                padding=(0, 0), bias=False
        )
        self.br1_bnorm1 = nn.BatchNorm2d(self.features[2])

        # branch 2: (1x1) conv bottleneck -> (3x3) conv
        self.br2_conv1 = nn.Conv2d(
                self.features[0], self.features[1], (1, 1), stride=1,
                padding=(0, 0), bias=False
        )
        self.br2_bnorm1 = nn.BatchNorm2d(self.features[1])
        self.br2_conv3 = nn.Conv2d(
                self.features[1], self.features[2], (3, 3), stride=1,
                padding=(1, 1), bias=False
        )
        self.br2_bnorm3 = nn.BatchNorm2d(self.features[2])

        # branch 3: (1x1) conv bottleneck -> (5x5) conv
        self.br3_conv1 = nn.Conv2d(
                self.features[0], self.features[1], (1, 1), stride=1,
                padding=(0, 0), bias=False
        )
        self.br3_bnorm1 = nn.BatchNorm2d(self.features[1])
        self.br3_conv5 = nn.Conv2d(
                self.features[1], self.features[2], (5, 5), stride=1,
                padding=(2, 2), bias=False
        )
        self.br3_bnorm5 = nn.BatchNorm2d(self.features[2])

        # branch 4: (3x3) avg pool (stride=1) -> (1x1) conv
        self.br4_avgpool = nn.AvgPool2d(3, stride=1, padding=1)
        self.br4_conv1 = nn.Conv2d(
                self.features[0], self.features[2], (1, 1), stride=1,
                padding=(0, 0), bias=False
        )
        self.br4_bnorm1 = nn.BatchNorm2d(self.features[2])

    def forward(self, X):
        X1 = self.br1_bnorm1(self.br1_conv1(X))
        X2 = self.br2_bnorm3(self.br2_conv3(
            F.relu(self.br2_bnorm1(self.br2_conv1(X)))
        ))
        X3 = self.br3_bnorm5(self.br3_conv5(
            F.relu(self.br3_bnorm1(self.br3_conv1(X)))
        ))
        X4 = self.br4_bnorm1(self.br4_conv1(self.br4_avgpool(X)))
        # concatenate on feature-map dimension (N x [C] x H x W)
        X = torch.cat([X1, X2, X3, X4], dim=1)
        return F.relu(X)


class InceptNet(nn.Module):

    def __init__(self, dims, K, block_features, pool_szs,
                 hidden_droprates, hidden_layer_sizes):
        super(InceptNet, self).__init__()
        # data shape
        self.dims = dims  # input dimensions
        self.K = K  # output classes
        # architecture
        self.block_features = block_features
        self.pool_szs = pool_szs
        self.drop_rates = hidden_droprates
        self.hidden_layer_sizes = hidden_layer_sizes
        # assemble network and move to GPU
        self.build()
        self.to(device)

    def build(self):
        # Inception Blocks
        self.blocks = nn.ModuleList()
        # fully connected layers
        self.dense_drops = nn.ModuleList()
        self.denses = nn.ModuleList()
        self.dense_bnorms = nn.ModuleList()

        for i, features in enumerate(self.block_features):
            self.blocks.append(InceptionBlock(features))

        # transform featuremaps into 1D vectors for transition to dense layers
        self.flatten = Flatten()

        # calculate dimensions going in to dense layers (post flattening len)
        # out features of last inception block times four (4 branch concat)
        num_fmaps = self.block_features[-1][2]*4
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
                nn.Dropout(p=self.drop_rates[i]))
            # fully-connected dense layer (linear transform)
            self.denses.append(nn.Linear(M1, M2, bias=False))
            # batch-normalization preceding non-linearity
            self.dense_bnorms.append(
                nn.BatchNorm1d(M2, affine=True, track_running_stats=True)
            )
            M1 = M2

        # output layers (logistic regression)
        self.log_drop = nn.Dropout(p=self.drop_rates[-1])
        self.logistic = nn.Linear(M1, self.K)

    def forward(self, X):
        # inception blocks
        for i, block in enumerate(self.blocks):
            X = block(X)
            if self.pool_szs[i] > 1:
                X = F.max_pool2d(X, kernel_size=self.pool_szs[i])
        # flatten before fully-connected layers
        X = self.flatten(X)
        # fully connected layers
        for drop, dense, bnorm in zip(
                self.dense_drops, self.denses, self.dense_bnorms):
            X = F.relu(bnorm(dense(X)))
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

            self.loss = nn.CrossEntropyLoss(reduction='mean').to(device)
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
                        train_acc = self.score(Xtrain[:1000], Ttrain[:1000])
                        test_cost, test_acc = self.cost_and_score(
                            Xtest[:1000], Ttest[:1000])
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
            plt.legend()
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


def inception_setup_1():
    dream = InceptNet(
        [28, 28], 10,  # input dimesions and number of output classes
        [[1, 32, 64], [256, 48, 64], [256, 48, 64]],  # inception blocks
        [2, 2, 2],  # pool sizes (after each block)
        [0],  # dropout rates (dense layers)
        [],  # fully connected layers (before logistic layer)
    )
    return dream


def inception_setup_2():
    dream = InceptNet(
        [28, 28], 10,  # input dimesions and number of output classes
        [[1, 32, 64], [256, 48, 64], [256, 48, 64], [256, 48, 64]],  # blocks
        [1, 2, 2, 2],  # pool sizes (after each block)
        [0],  # dropout rates (dense layers)
        [],  # fully connected layers (before logistic layer)
    )
    return dream


def main():
    X, T = loadAndProcess()
    Xtrain, Ttrain, Xtest, Ttest = trainTestSplit(X, T, ratio=.8)
    del X, T  # free up memory
    incept = inception_setup_2()
    incept.fit(Xtrain, Ttrain, Xtest, Ttest, lr=1e-3, epochs=25, batch_sz=100)


if __name__ == '__main__':
    main()
