import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
import string
import os.path

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
    Note the modifications to the operations done before loss (predictions),
    not taking the, but outputing K 0->1 non-softmax predictions.

    This example is working with the Kaggle Toxic Comment dataset.
    """
    def __init__(self, V, K, conv_layer_shapes, pool_szs, hidden_layer_sizes,
                 p_drop, embeddings=None, batch_mu=.1, epsilon=1e-4):
        super(LanguageCNN, self).__init__()
        # data shape
        self.V = V  # vocab size (not actually, more like largest word idx)
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
        self.build(embeddings)
        self.to(device)

    def build(self, embeddings):
        if embeddings is not None:
            self.embed = nn.Embedding.from_pretrained(
                torch.from_numpy(embeddings).float(),
                freeze=True
            )
        else:
            self.embed = nn.Embedding(self.V, 100, padding_idx=0)

        # convolutional layers (MaxPool and ELU applied in forward())
        self.convs = nn.ModuleList()
        self.conv_bnorms = nn.ModuleList()
        # fully connected layers
        self.dense_drops = nn.ModuleList()
        self.denses = nn.ModuleList()
        self.dense_bnorms = nn.ModuleList()

        for i, shape in enumerate(self.conv_layer_shapes):
            # 1D convolutional layers (in features, out features, kernel)
            self.convs.append(
                nn.Conv1d(shape[1], shape[2], shape[0], stride=1,
                          padding=shape[0]//2, bias=False)
            )
            # batch normalization (pass through before non-linearity)
            self.conv_bnorms.append(nn.BatchNorm1d(shape[2]))

        # transform conv output into 1D vectors for transition to dense layers
        self.flatten = Flatten()

        # input size to first dense layer (global pooled conv output)
        M1 = self.conv_layer_shapes[-1][2]
        for i, M2 in enumerate(self.hidden_layer_sizes):
            # dropout preceding fully-connected layer
            self.dense_drops.append(nn.Dropout(p=self.drop_rates[i]))
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
        # convert sequence of indices to word-vector matrices
        X = self.embed(X).transpose(1, 2)  # switch time to last dimension
        # convolutional layers
        for i, (conv, bnorm) in enumerate(zip(self.convs, self.conv_bnorms)):
            X = F.elu(bnorm(conv(X)))
            if self.pool_szs[i] > 1:
                X = F.max_pool1d(X, kernel_size=self.pool_szs[i])
        # pool over all channels and flatten (input to dense is len C vector)
        X = self.flatten(F.adaptive_max_pool1d(X, 1))  # max outperforms avg
        # fully connected layers
        for drop, dense, bnorm in zip(
                self.dense_drops, self.denses, self.dense_bnorms):
            X = F.elu(bnorm(dense(X)))
        # pass logits through sigmoid (each output is binary classification)
        return torch.sigmoid(self.logistic(self.log_drop(X)))

    def fit(self, Xtrain, Ttrain, Xtest, Ttest, lr=1e-4, epochs=40,
            batch_sz=100, print_every=50):

            N = Xtrain.shape[0]  # number of samples

            # send data to GPU
            Xtrain = torch.from_numpy(Xtrain).long().to(device)
            Ttrain = torch.from_numpy(Ttrain).float().to(device)
            Xtest = torch.from_numpy(Xtest).long().to(device)
            Ttest = torch.from_numpy(Ttest).float().to(device)

            self.loss = nn.BCELoss(reduction='mean').to(device)
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
                        train_acc = self.score(
                            Xtrain[:batch_sz*5], Ttrain[:batch_sz*5])
                        test_cost, test_acc = self.cost_and_score(
                            Xtest[:batch_sz*5], Ttest[:batch_sz*5])
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
        return logits.data.cpu().numpy().round()

    def score(self, inputs, labels):
        predictions = self.predict(inputs)
        return np.mean(labels.cpu().numpy() == predictions)

    def cost_and_score(self, inputs, labels):
        self.eval()  # set the model to testing mode
        with torch.no_grad():
            # Forward
            logits = self.forward(inputs)
            output = self.loss.forward(logits, labels)
        predictions = logits.data.cpu().numpy().round()
        acc = np.mean(labels.cpu().numpy() == predictions)
        return output.item(), acc


def tokenizer(s):
    "Remove puncutation, downcase and split on spaces and return a list"
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = s.lower()  # downcase
    return s.split()


def process(comments):
    """
    Take in list of comments (strings) and tokenize them to build word index
    mappings to allowing conversion of comments into word vectors. Outputs are
    sequences of indices (used to map words to embeddings), the word->index
    and index->word) mappings and the frequencies of each word. The
    frequencies can be used to trim down the vocabulary to the most common
    words if the original size is too great.
    """
    word_index_map = {'PAD_TOKEN': 0}
    current_index = 1  # 0 is reserved for padding
    sequences = []
    index_word_map = ['PAD_TOKEN']
    freqs = {}
    for comment in comments:
        sequence = []
        tokens = tokenizer(comment)
        for token in tokens:
            if token not in word_index_map:
                word_index_map[token] = current_index
                current_index += 1
                index_word_map.append(token)
                freqs[token] = 1
            else:
                freqs[token] += 1
            sequence.append(word_index_map[token])
        sequences.append(sequence)

    return sequences, word_index_map, index_word_map, freqs


def trim_vocab(seqs, word2idx, freqs, MAX_VOCAB=20000, padNum=0):
    """
    Sort words by descending frequency and keep the top MAX_VOCAB of them. All
    other word indices (not in common set) are written over with a padding
    index (padNum, default: 0).
    """
    words, idxs = [], []
    for k in sorted(freqs, key=freqs.get, reverse=True)[:MAX_VOCAB]:
        words.append(k)
        idxs.append(word2idx[k])

    # replace all sequence elements not in most common set with padding tokens
    common = set(idxs)  # convert to set for quick lookup (vs list)
    seqs = [[idx if idx in common else padNum for idx in seq] for seq in seqs]
    return seqs, words, idxs


def pad_seqs(sequences, padNum=0):
    """
    Take in list of lists (sequences of variable length), and pad them with
    a given number (default: 0) to the same length (longest in sequences).
    """
    max_len = 0
    for i, seq in enumerate(sequences):
        max_len = len(seq) if len(seq) > max_len else max_len

    sequences = [seq + [padNum]*(max_len - len(seq)) for seq in sequences]
    return sequences


def get_embeddings(filestr, common_idxs, common_words):
    # immediately transpose, so columns/keys are the words
    df = pd.read_csv(filestr, index_col=0, header=None, sep=' ', quoting=3).T
    embed_dim = df.shape[0]  # number of emebdding dimensions
    embed = np.zeros((max(common_idxs)+1, embed_dim))  # to be filled

    # set-up a progress bar since this takes a bit
    tick = np.floor(len(common_words)/50)  # for progress bar (2% each)
    print('Importing pre-trained word-embeddings...')
    print('['+' '*50+']', end='\r', flush=True)  # return allows overwritting

    # fill in rows of embedding matrix with (trimmed) vocabulary
    for prog, (idx, word) in enumerate(zip(common_idxs, common_words), 1):
        embed[idx, :] = df.get(word, default=1e-6)  # try not zero, embed kills
        if prog % tick == 0:
            # overwrite progress bar with an additional tick
            ticks = int(np.floor(prog/tick))
            print('[' + '='*ticks + ' '*(50-ticks) + ']', end='\r', flush=True)
    print('')  # newline
    return embed


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
    # https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
    toxic_data = pd.read_csv(datapath+'toxic_comments/train.csv')

    # pull out targets
    categories = ['toxic', 'severe_toxic', 'obscene', 'threat',
                  'insult', 'identity_hate']
    labels = toxic_data[categories].values
    K = labels.shape[1]

    # pull out comment text data
    comments = toxic_data['comment_text'].values
    del toxic_data  # don't need dataframe anymore
    seqs, word2idx, idx2word, freqs = process(comments)

    # process sequence data for the CNN (limit vocab and pad to same length)
    seqs, common_words, common_idxs = trim_vocab(seqs, word2idx, freqs)
    V = max(common_idxs)  # not really the vocab size (just a quick fix)
    print('Total vocabulary size:', len(idx2word))
    print('Trimmed vocabulary size:', V)

    # pad sequences to same length and convert to a 2D numpy matrix (N x Time)
    X = np.array(pad_seqs(seqs))
    print('X shape:', X.shape)
    print('T shape:', labels.shape)

    # load and process pre-trained embeddings
    # http://nlp.stanford.edu/data/glove.6B.zip
    if os.path.isfile(datapath+'/glove_embeddings/glove100_toxic.npy'):
        embeds = np.load(datapath+'/glove_embeddings/glove100_toxic.npy')
    else:
        embeds = get_embeddings(
            datapath+'/glove_embeddings/glove.6B.100d.txt',
            common_idxs, common_words
        )
        np.save(datapath+'/glove_embeddings/glove100_toxic.npy', embeds)
    print('Word Embeddings shape:', embeds.shape)

    # train-test split of sequence matrices (X) and sequence labels
    Xtrain, Ttrain, Xtest, Ttest = trainTestSplit(X, labels, ratio=.8)

    # build CNN and fit to data
    cnn = LanguageCNN(
        V, K,
        [[3, 100, 32], [3, 32, 64], [3, 64, 128]],  # conv1d (kernel, in, out)
        [2, 2, 0],  # pooling
        [256],  # dense layers
        [.5, .5],  # dropout
        embeddings=embeds
    )
    cnn.fit(
        Xtrain, Ttrain, Xtest, Ttest,
        epochs=5, batch_sz=200, print_every=100
    )


if __name__ == '__main__':
    datapath = 'C:/Users/geoff/GitRepos/learning/large_files/'
    main()
