import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch import optim

# use GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


class ANN(object):

    def __init__(self, hidden_layer_sizes, p_drop):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_rates = p_drop

    def fit(self, train_set, test_set, lr=1e-4, batch_mu=.1,
            epsilon=1e-4, epochs=40, batch_sz=200, print_every=50):

            train_loader = DataLoader(train_set, batch_size=batch_sz,
                                      shuffle=True)
            test_loader = DataLoader(test_set, batch_size=batch_sz)

            D, K = train_set.D, train_set.K

            self.model = torch.nn.Sequential()
            M1 = D  # first input layer is the number of features in X
            for i, M2 in enumerate(self.hidden_layer_sizes):
                self.model.add_module(
                    "dropout"+str(i+1),
                    torch.nn.Dropout(p=self.dropout_rates[i])
                )
                self.model.add_module(
                    "dense"+str(i+1),
                    torch.nn.Linear(M1, M2)
                )
                self.model.add_module(
                    "batchnorm"+str(i+1),
                    torch.nn.BatchNorm1d(
                        M2, eps=epsilon, momentum=batch_mu, affine=True,
                        track_running_stats=True
                    )
                )
                self.model.add_module(
                    "relu"+str(i+1),
                    torch.nn.ReLU()
                )
                M1 = M2  # input layer to next layer is this layer
            # output layer (no activation)
            self.model.add_module(
                "dropoutOut",
                torch.nn.Dropout(p=self.dropout_rates[-1])
            )
            self.model.add_module(
                "denseOut",
                torch.nn.Linear(M1, K)
            )

            self.model.to(device)

            self.loss = torch.nn.CrossEntropyLoss(size_average=True)
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

            train_costs, train_accs, test_costs, test_accs = [], [], [], []
            for i in range(epochs):
                cost = 0
                print("epoch:", i)
                for j, batch in enumerate(train_loader):

                    cost += self.train(batch['face'], batch['emotion'])

                    if j % print_every == 0:
                        train_acc = self.score(batch['face'], batch['emotion'])
                        test_acc, test_cost = 0, 0
                        for t, testB in enumerate(test_loader, 1):
                            test_acc += self.score(
                                testB['face'], testB['emotion']
                            )
                            test_cost += self.get_cost(
                                testB['face'], testB['emotion']
                            )
                        test_acc /= (t+1)
                        test_cost /= (t+1)
                        print("cost: %f, acc: %.2f" % (test_cost, test_acc))

                # for plotting
                train_costs.append(cost / (j+1))
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

    df = pd.DataFrame({'face': np.concatenate(Xlist, axis=0),
                      'emotion': np.concatenate(Tlist, axis=0)})
    return df


class FaceRecog(Dataset):
    '''Kaggle Facial Recognition Dataset'''

    def __init__(self, df):
        self.df = df
        self.K = pd.unique(df['emotion']).size
        self.D = len(df['face'][0].split(' '))
        self.faces = np.array(
            [str.split(' ') for str in df['face'].values]
        ).astype(np.uint8)
        self.emotions = df['emotion'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        face = torch.from_numpy(self.faces[idx]).float().to(device)
        emotion = torch.from_numpy(np.array(self.emotions[idx])).to(device)
        # formatting each one before is much slower than doing it all at once
        # repeating the same processing over and over again.
        # face = torch.from_numpy(
        #     np.array(self.df['face'][idx].split(' ')).astype(np.uint8)
        # ).float().to(device)
        # emotion = torch.from_numpy(
        #     np.array(self.df['emotion'][idx])
        # ).long().to(device)
        sample = {'face': face, 'emotion': emotion}
        return sample


def main():
    print('loading in data...')
    df = pd.read_csv('fer2013.csv')
    print('data loaded.')
    print('samples in full dataset:', df['emotion'].values.size)
    df = df.loc[:10000]
    df = classRebalance(df['pixels'], df['emotion'])
    print('emotion counts:',
          [(df['emotion'] == k).sum() for k in np.unique(df['emotion'])])
    df_train, df_test = train_test_split(df, test_size=0.2)
    df_train, df_test = df_train.reset_index(), df_test.reset_index()
    train_set, test_set = FaceRecog(df_train), FaceRecog(df_test)

    ann = ANN([1000, 1000, 500, 500, 300, 100],
              [0.2, 0.5, 0.5, .5, .5, .5, .5])
    ann.fit(train_set, test_set, lr=1e-3, epochs=100)


if __name__ == '__main__':
    main()
