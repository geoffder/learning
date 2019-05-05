import numpy as np
import matplotlib.pyplot as plt
# from sklearn.utils import shuffle

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


class LanguageCNN(nn.Module):
    """
    Input sequences of words, which are then one-hot encoded, and padded (so
    all sequences are the same length) then converted to word-vectors with
    pre-trained embeddings. Sequences of vectors are then subject to 1D
    convolutions before dense layers and K binary classifications are done.

    This example is working with the Kaggle Toxic Comment dataset.
    """
    def __init__(self, dims, K, conv_layer_shapes, pool_szs,
                 hidden_layer_sizes, p_drop, batch_mu=.1, epsilon=1e-4):
        super(LanguageCNN, self).__init__()
        # data shape
        self.dims = dims  # input dimensions
        self.K = K  # output classes
        # conv architecture
        self.conv_layer_shapes = conv_layer_shapes
        self.pool_szs = pool_szs
        # dense architecture
        self.hidden_layer_sizes = hidden_layer_sizes
        self.drop_rates = p_drop
        # batch-norm hyper-parameters
        self.epsilon = epsilon
        self.batch_mu = batch_mu
        # assemble network and move to GPU
        self.build()
        self.to(device)

    def build(self):
        # TODO: Add word embedding layer that uses frozen pre-trained weights
        # TODO: Change convolutional layers to do 1D convolution
        # convolutional layers (MaxPool and ELU applied in forward())
        self.convs = nn.ModuleList()
        self.conv_bnorms = nn.ModuleList()
        # fully connected layers
        self.dense_drops = nn.ModuleList()
        self.denses = nn.ModuleList()
        self.dense_bnorms = nn.ModuleList()

        for i, shape in enumerate(self.conv_layer_shapes):
            # convolutional layers (in features, out features, (H dim, W dim))
            self.convs.append(
                nn.Conv2d(shape[2], shape[3], (shape[0], shape[1]), stride=1,
                          padding=(shape[0]//2, shape[1]//2), bias=False)
            )

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
            self.denses.append(nn.Linear(M1, M2, bias=False))
            # batch-normalization preceding non-linearity
            self.dense_bnorms.append(
                nn.BatchNorm1d(M2, eps=self.epsilon, momentum=self.batch_mu,
                               affine=True, track_running_stats=True)
            )
            M1 = M2

        # output layers (final dropout and logistic regression)
        self.log_drop = nn.Dropout(p=self.drop_rates[-1])
        self.logistic = nn.Linear(M1, self.K)

    def forward(self, X):
        # convolutional layers
        for i, (drop, conv, bnorm) in enumerate(zip(
                self.conv_drops, self.convs, self.conv_bnorms)):
            X = F.elu(bnorm(conv(drop(X))))
            if self.pool_szs[i] > 1:
                X = F.max_pool2d(X, kernel_size=self.pool_szs[i])
        X = self.flatten(X)
        # fully connected layers
        for drop, dense, bnorm in zip(
                self.dense_drops, self.denses, self.dense_bnorms):
            X = F.elu(bnorm(dense(X)))
        # get logits
        # TODO: Change to use sigmoid independently on each of the K outputs
        # TODO: Need binary pred for each label independently (can be multiple)
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


def main():
    pass


if __name__ == '__main__':
    main()
