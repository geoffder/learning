import numpy as np
import pandas as pd
from skimage.transform import resize
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

import torch
from torch import nn
from torch import optim
from torchvision import models

# use GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


class TransferResNet50(object):
    """
    Trying out tranfer learning with ResNet50. Using the same structure
    I have used for my own networks, but instead uses a pre-trained build of
    the VGG network with the relevant components switched out to fit the data.
    """
    def __init__(self, dims=None, classes=1000, pretrained=True,
                 freeze_network=False):
        # only specify input dimensions if not [3, 244, 244] (ImageNet dims)
        self.dims = dims  # input dimensions
        self.K = classes  # output classes
        self.pretrained = pretrained
        self.freeze_network = freeze_network
        self.model = self.build()
        self.model.to(device)

    def build(self):
        resnet = models.resnet50(pretrained=self.pretrained)

        # freeze covolutional layers
        if self.freeze_network:
            for param in resnet.parameters():
                param.requires_grad = False

        if self.dims is not None:
            # change to appropriate number of input feature-maps
            krnl = [3, 3]
            resnet.conv1 = nn.Conv2d(
                self.dims[0], 64, kernel_size=(krnl[0], krnl[1]),
                stride=(1, 1), padding=(krnl[0]//2, krnl[1]//2), bias=False
            )
            resnet.maxpool = nn.MaxPool2d(kernel_size=2)  # small images

        # change to appropriate number of output classes
        resnet.avgpool = nn.AvgPool2d(kernel_size=2)  # reduce for fashion
        resnet.fc = nn.Linear(
            in_features=2048, out_features=self.K, bias=True)

        return resnet

    def fit(self, Xtrain, Ttrain, Xtest, Ttest, lr=1e-4, epochs=40,
            batch_sz=200, print_every=50):

            N = Xtrain.shape[0]  # number of samples

            # send labels to GPU (data will be sent in batches)
            Ttrain = torch.from_numpy(Ttrain).long().to(device)
            Ttest = torch.from_numpy(Ttest).long().to(device)

            self.loss = nn.CrossEntropyLoss(reduction='mean').to(device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

            n_batches = N // batch_sz
            train_costs, train_accs, test_costs, test_accs = [], [], [], []
            for i in range(epochs):
                cost = 0
                print("epoch:", i, "n_batches:", n_batches)
                # shuffle dataset for next epoch of batches
                inds = torch.randperm(Xtrain.shape[0])
                Xtrain, Ttrain = Xtrain[inds], Ttrain[inds]
                for j in range(n_batches):
                    Xbatch = Xtrain[j*batch_sz:(j*batch_sz+batch_sz)]
                    # re-size batch (enlarge) to match expected input
                    Xbatch = torch.from_numpy(
                        resize_batch(Xbatch, sz=self.dims[1])
                    ).float().to(device)
                    Tbatch = Ttrain[j*batch_sz:(j*batch_sz+batch_sz)]

                    cost += self.train_step(Xbatch, Tbatch)

                    if j % print_every == 0:
                        # shuffle test set
                        inds = torch.randperm(Xtest.shape[0])
                        Xtest, Ttest = Xtest[inds], Ttest[inds]
                        # re-size batch (enlarge) to match expected input
                        Xtest_batch = torch.from_numpy(
                            resize_batch(Xtest[:batch_sz], sz=self.dims[1])
                        ).float().to(device)
                        Ttest_batch = Ttest[:batch_sz]
                        # accuracies for train and test sets
                        train_acc = self.score(Xbatch, Tbatch)
                        test_cost, test_acc = self.cost_and_score(
                            Xtest_batch, Ttest_batch)
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
        self.model.train()  # set the model to training mode
        self.optimizer.zero_grad()  # Reset gradient

        # Forward
        logits = self.model.forward(inputs)
        output = self.loss.forward(logits, labels)

        # Backward
        output.backward()
        self.optimizer.step()  # Update parameters

        return output.item()

    def get_cost(self, inputs, labels):
        self.model.eval()  # set the model to testing mode
        self.optimizer.zero_grad()  # Reset gradient
        with torch.no_grad():
            # Forward
            logits = self.model.forward(inputs)
            output = self.loss.forward(logits, labels)
        return output.item()

    def predict(self, inputs):
        self.model.eval()
        self.optimizer.zero_grad()  # Reset gradient
        with torch.no_grad():
            logits = self.model.forward(inputs)
        return logits.data.cpu().numpy().argmax(axis=1)

    def score(self, inputs, labels):
        predictions = self.predict(inputs)
        return np.mean(labels.cpu().numpy() == predictions)

    def cost_and_score(self, inputs, labels):
        self.model.eval()  # set the model to testing mode
        self.optimizer.zero_grad()  # Reset gradient
        with torch.no_grad():
            # Forward
            logits = self.model.forward(inputs)
            output = self.loss.forward(logits, labels)
        predictions = logits.data.cpu().numpy().argmax(axis=1)
        acc = np.mean(labels.cpu().numpy() == predictions)
        return output.item(), acc


def resize_batch(batch, sz=244):
    X = np.zeros((batch.shape[0], 1, sz, sz))  # N x H x W x C
    for i in range(batch.shape[0]):
        X[i, 0, :, :] = resize(batch[i], (sz, sz), mode='constant',
                               anti_aliasing=True)
    return X


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

    X = data[:, 1:].reshape(data.shape[0], 28, 28) / 255
    T = data[:, 0]  # labels are first column
    print('X shape:', X.shape)
    print('T shape:', T.shape)

    return X, T


def main():
    X, T = loadAndProcess()
    Xtrain, Ttrain, Xtest, Ttest = trainTestSplit(X, T, ratio=.8)
    del X, T  # free up memory
    resn = TransferResNet50(
        dims=[1, 28, 28], classes=10, pretrained=False, freeze_network=False
    )
    resn.fit(Xtrain, Ttrain, Xtest, Ttest, lr=1e-3, epochs=10, batch_sz=75)


if __name__ == '__main__':
    main()
