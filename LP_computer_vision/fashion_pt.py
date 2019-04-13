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
                 hidden_layer_sizes, p_drop, epsilon, batch_mu):
        self.dims = dims  # input dimensions
        self.K = K  # output classes
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
        self.conv_drops = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.maxpools = nn.ModuleList()
        self.conv_bnorms = nn.ModuleList()

        self.dense_drops = nn.ModuleList()
        self.denses = nn.ModuleList()
        self.dense_bnorms = nn.ModuleList()

        for i, shape in enumerate(self.conv_layer_shapes):
            self.conv_drops.append(nn.DropOut2d(p=self.drop_rates[i]))

            self.convs.append(
                nn.Conv2d(shape[2], shape[3], (shape[0], shape[1]), stride=1,
                          padding=(shape[0]//2, shape[1]//2))
            )

            # stride for pool is same as kernel size by default
            self.maxpools.append(nn.MaxPool2d(self.pool_szs[i]))

            self.conv_bnorms.append(nn.BatchNorm2d(shape[3]))

            # do ELU in forward, don't bother with module. Consider same for
            # max pooling and dropouts. No point in modules really.

        self.flatten = Flatten()

        # calculate dimensions going in to dense layers
        num_fmaps = self.conv_layer_shapes[-1][3]
        pool_redux = np.prod([p for p in self.pool_szs])
        D = self.dims[0]//pool_redux * self.dims[1]//pool_redux * num_fmaps

        M1 = D  # first input layer is the number of features in X
        for i, M2 in enumerate(self.hidden_layer_sizes):
            self.dense_drops.append(
                nn.Dropout(p=self.drop_rates[len(self.conv_drops)+i]))

            self.denses.append(nn.Linear(M1, M2))

            self.dense_bnorms.append(
                nn.BatchNorm1d(M2, eps=self.epsilon, momentum=self.batch_mu,
                               affine=True, track_running_stats=True)
            )
            M1 = M2

        # output layers
        self.dense_drops.append(nn.Dropout(p=self.drop_rates[-1]))
        self.logistic = torch.nn.Linear(M1, self.K)

    def fit(self, Xtrain, Ttrain, Xtest, Ttest, lr=1e-4, epochs=40,
            batch_sz=200, print_every=50):

            N = Xtrain.shape[0]

            Xtrain = torch.from_numpy(Xtrain).float().to(device)
            Ttrain = torch.from_numpy(Ttrain).long().to(device)
            Xtest = torch.from_numpy(Xtest).float().to(device)
            Ttest = torch.from_numpy(Ttest).long().to(device)

            self.loss = torch.nn.CrossEntropyLoss(size_average=True).to(device)
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
                        inds = torch.randperm(Xtest.size()[0])
                        Xtest, Ttest = Xtest[inds], Ttest[inds]
                        train_acc = self.score(Xtrain[:1000], Ttrain[:1000])
                        test_cost, test_acc = self.cost_and_score(
                            Xtest[:1000], Ttest[:1000])
                        print("cost: %f, acc: %.2f" % (test_cost, test_acc))

                # for plotting
                train_costs.append(cost / n_batches)
                train_accs.append(train_acc)
                test_costs.append(test_cost)
                test_accs.append(test_acc)

            plt.plot(train_costs, label='training cost')
            plt.plot(test_costs, label='validation cost')
            plt.legend()
            plt.show()
            plt.plot(train_accs, label='training accuracy')
            plt.plot(test_accs, label='validation accuracy')
            plt.legend()
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


def classRebalance(X, T):
    '''
    Take data and labels and increase number of samples for under-represented
    classes by duplicating the existing ones.
    '''
    classes = np.unique(T)
    Xlist = [X[T == k] for k in classes]
    Tlist = [T[T == k] for k in classes]
    bigN = np.max([t.shape for t in Tlist])
    # develop a less coarse way that better approximates the classes
    Xlist = [np.concatenate([x]*(bigN//x.shape[0]), axis=0) for x in Xlist]
    Tlist = [np.concatenate([t]*(bigN//t.shape[0]), axis=0) for t in Tlist]

    return np.concatenate(Xlist, axis=0), np.concatenate(Tlist, axis=0)


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

    df = pd.read_csv('fer2013.csv')
    print('samples in full dataset:', df['emotion'].values.size)
    maxN = 20000
    print('taking %d samples' % maxN)
    X = np.array(
        [str.split(' ') for str in df['pixels'].values[:maxN]]
    ).astype(np.uint8)
    T = df['emotion'].values[:maxN]
    print('data loaded.')

    faces = np.zeros((X.shape[0], 1, 48, 48))  # one channel, B&W img
    for i, img in enumerate(X):
        # dimensions in pytorch: N, C, H, W
        faces[i, 0, :, :] = img.reshape(48, 48) / 255
    X, T = classRebalance(faces, T)
    print('X shape:', X.shape, 'T shape:', T.shape)
    print('emotion counts:', [(T == k).sum() for k in np.unique(T)])
    return X, T


def main():
    X, T = loadAndProcess()
    Xtrain, Ttrain, Xtest, Ttest = trainTestSplit(X, T, ratio=.8)
    del X, T  # free up memory
    ann = CNN([[5, 5, 1, 20], [5, 5, 20, 50], [5, 5, 50, 50]],
              [2, 2, 2],
              [1000, 1000, 500, 500, 300, 100],
              [0.2, 0.5, 0.5, .5, .5, .5, .5])
    ann.fit(Xtrain, Ttrain, Xtest, Ttest, lr=1e-3, epochs=30)


if __name__ == '__main__':
    main()
