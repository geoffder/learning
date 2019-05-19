import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
import string
import os.path

import torch
from torch import nn
from torch import optim


# use GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


class Seq2Seq(nn.Module):
    """
    Sequence-to-sequence machine translation model.
    """
    def __init__(self, in_len, targ_len, in_V, targ_V, latent_dim_size,
                 num_LSTM_layers, embeddings=None, LSTM_drop=0):
        super(Seq2Seq, self).__init__()
        # data shape
        self.in_len = in_len
        self.targ_len = targ_len
        self.in_V = in_V  # input vocabulary size
        self.targ_V = targ_V  # target vocabulary size
        # architecture
        self.latent_dim_size = latent_dim_size  # same for encoder and decoder
        self.num_LSTM_layers = num_LSTM_layers
        self.LSTM_drop = LSTM_drop
        # assemble network and move to GPU
        self.build(embeddings)
        self.to(device)

    def build(self, embeddings):
        # encoder word embeddings (pre-trained or randomly initialized)
        if embeddings is not None:
            self.encoder_embed = nn.Embedding.from_pretrained(
                torch.from_numpy(embeddings).float(),
                freeze=True
            )
        else:
            self.encoder_embed = nn.Embedding(self.in_V, 100, padding_idx=0)
        # decoder network word embeddings (randomly initialized)
        self.decoder_embed = nn.Embedding(self.targ_V, 100, padding_idx=0)

        # encoder and decoder recurrent networks
        self.encoder_LSTM = nn.LSTM(
            100, self.latent_dim_size, self.num_LSTM_layers,
            bias=True, dropout=self.LSTM_drop, batch_first=True
        )
        self.decoder_LSTM = nn.LSTM(
            100, self.latent_dim_size, self.num_LSTM_layers,
            bias=True, dropout=self.LSTM_drop, batch_first=True
        )

        # fully connected layers for decoder -> word probabilities
        self.logistic = nn.Linear(self.latent_dim_size, self.targ_V)

    def forced_teaching(self, X, T):
        """
        Takes both input and target (offset by <sos> token) sequences, as the
        offset targets are fed in to the decoder network, rather than the
        decoder's own predictions (as they would be during generation). This
        ensures that the model is able to learn the entire sequence during
        every pass, rather than de-railing and training on garbage after one
        false prediction.
        """
        # convert sequences of indices to word-vector matrices
        X = self.encoder_embed(X)  # shape: (batch, T, D)
        T = self.decoder_embed(T)

        _, thought = self.encoder_LSTM(X)
        output, _ = self.decoder_LSTM(T, thought)

        # get logits
        return self.logistic(output)

    def translate(self, X):
        """
        Forward pass without teacher forcing. Model will predict the next word
        of the output based on the translation SO FAR, regardless of whether
        it is correct or not. Used for testing during fitting, and also for
        translating sequences after training is over.
        """
        out = np.zeros((X.shape[0], self.targ_len))
        for i in range(X.shape[0]):
            _, (h, c) = self.encoder_LSTM(self.encoder_embed(X[i].view(1, -1)))
            seq = [self.targ_w2i['<sos>']]
            for t in range(self.targ_len):
                vec = self.decoder_embed(
                    torch.LongTensor([[seq[-1]]]).to(device)
                )
                _, (h, c) = self.decoder_LSTM(vec, (h, c))

                pred = torch.argmax(self.logistic(h[-1])).long()
                seq.append(pred.detach().cpu().tolist())
                if seq[-1] == self.targ_w2i['<eos>']:
                    # print([self.targ_i2w[idx] for idx in seq])
                    break

            seq = seq[1:] + [self.targ_w2i['<pad>']]*(self.targ_len-t-1)
            out[i, :] = np.array(seq)  # ignore <sos> token
        return torch.from_numpy(out).long().to(device)

    def fit(self, inputs, targets, forced_inputs, targ_w2i, targ_i2w,
            lr=1e-4, epochs=40, batch_sz=200, print_every=50):

            N = inputs.shape[0]  # number of samples
            self.targ_w2i = targ_w2i  # word to index for target vocabulary
            self.targ_i2w = targ_i2w  # index to word for target vocabulary

            # send data to GPU
            inputs = torch.from_numpy(inputs).long().to(device)
            targets = torch.from_numpy(targets).long().to(device)
            forced = torch.from_numpy(forced_inputs).long().to(device)

            self.loss = nn.CrossEntropyLoss(
                reduction='sum', ignore_index=0).to(device)
            # self.loss = nn.CrossEntropyLoss(reduction='mean').to(device)
            self.optimizer = optim.Adam(self.parameters(), lr=lr)

            n_batches = N // batch_sz
            train_costs, train_accs, test_accs = [], [], []
            for i in range(epochs):
                cost = 0
                print("epoch:", i, "n_batches:", n_batches)
                # shuffle dataset for next epoch of batches
                inds = torch.randperm(inputs.size()[0])
                inputs, targets = inputs[inds], targets[inds]
                forced = forced[inds]
                for j in range(n_batches):
                    Xbatch = inputs[j*batch_sz:(j*batch_sz+batch_sz)]
                    Tbatch = targets[j*batch_sz:(j*batch_sz+batch_sz)]
                    Fbatch = forced[j*batch_sz:(j*batch_sz+batch_sz)]

                    cost += self.train_step(Xbatch, Tbatch, Fbatch)

                    if j % print_every == 0:
                        # cost and accuracy during forced teaching
                        train_cost, train_acc = self.train_score(
                            Xbatch, Tbatch, Fbatch
                        )
                        print(
                            'train cost: %.2f, train acc: %.3f'
                            % (train_cost, train_acc)
                        )
                        # accuracy during 'free' translation (non-forced)
                        test_acc = self.trans_score(Xbatch, Tbatch)
                        print("test accuracy: %.3f" % (test_acc))

                # for plotting
                train_costs.append(cost / n_batches)
                train_accs.append(train_acc)
                test_accs.append(test_acc)

            # plot cost and accuracy progression
            fig, axes = plt.subplots(1, 2)
            axes[0].plot(train_costs, label='training')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Cost')
            axes[1].plot(train_accs, label='forced teaching')
            axes[1].plot(test_accs, label='translation')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            plt.legend()
            fig.tight_layout()
            plt.show()

    def train_step(self, inputs, targets, forced_inputs):
        self.train()  # set the model to training mode
        self.optimizer.zero_grad()  # Reset gradient

        # Forward
        logits = self.forced_teaching(inputs, forced_inputs)
        # output = self.loss.forward(
        #     logits.view(-1, logits.shape[2]), targets.view(-1))
        output = self.loss.forward(logits.transpose(2, 1), targets)

        # Backward
        output.backward()
        self.optimizer.step()  # Update parameters

        return output.item()

    def train_score(self, inputs, targets, forced_inputs, pad_corr=False):
        """
        Get cost and accuracy of a forced_teaching run. Measure of how well
        model is able to predict the next word of the translated sequence,
        using the correct sequence as the input, regardless of the actual
        prediction.
        """
        # forward pass using forced teaching
        self.eval()  # set the model to testing mode
        with torch.no_grad():
            logits = self.forced_teaching(inputs, forced_inputs)
            output = self.loss.forward(logits.transpose(2, 1), targets)

        # torch => numpy
        labels = targets.cpu().numpy()
        predictions = torch.argmax(logits, dim=2).long().cpu().numpy()
        labels[labels == 0] = 2  # predictions are "padded" with 2 (<eos>)

        if pad_corr:
            # estimate accuracy with proportion of padding tokens removed
            pads = labels.size - np.count_nonzero(targets)
            acc = (np.sum(labels == predictions) - pads) / (labels.size - pads)
        else:
            acc = np.mean(labels == predictions)

        # cost and accuracy
        return output.item(), acc

    def predict(self, inputs):
        "Forward pass using free-translation (not teacher forced)."
        self.eval()
        with torch.no_grad():
            seqs = self.translate(inputs)
        return seqs.detach().cpu().numpy()

    def trans_score(self, inputs, labels, pad_corr=False):
        "Calculate accuracy of translation. Option to adjust for padding."
        predictions = self.predict(inputs)
        labels = labels.cpu().numpy()
        if pad_corr:
            pads = labels.size - np.count_nonzero(labels)
            acc = (np.sum(labels == predictions) - pads) / (labels.size - pads)
        else:
            acc = np.mean(labels == predictions)
        return acc

    def demo(self, sequences, in_i2w):
        """
        Take in inputs sequences (index representation of 'from' language),
        translate in to target language and display results. For checking how
        well model the model has fit the machine translation task.
        """
        inputs = torch.from_numpy(sequences).long().to(device)
        while True:
            idx = np.random.randint(0, sequences.shape[0])
            out = self.translate(inputs[idx].view(1, -1))
            trans = [
                self.targ_i2w[i] if i > 2 else ''
                for i in out.detach().cpu().numpy().flatten()
            ]
            print(
                ' '.join([in_i2w[i] if i > 2 else '' for i in sequences[idx]]),
                end=' '
            )
            print('=> ' + ' '.join(trans), end='\n\n')
            again = input(
                "Show another translation? Enter 'n' to quit\n")
            if again == 'n':
                break


def load_samples(pth, num_samples):
    input_texts, target_texts, target_texts_inputs = [], [], []
    for i, line in enumerate(open(pth+'translation/fra.txt'), 1):
        # only keep a limited number of samples
        if i > num_samples:
            break

        # input and target are separated by tab
        if '\t' not in line:
            continue

        # split up the input and translation
        input_text, translation = line.rstrip().split('\t')
        input_texts.append(input_text)
        # target input used for teacher-forcing during training
        target_texts.append(translation + ' <eos>')
        target_texts_inputs.append('<sos> ' + translation)

    print("num samples:", len(input_texts))
    return input_texts, target_texts, target_texts_inputs


def tokenizer(s, keep_punc=False):
    "Remove puncutation, downcase and split on spaces and return a list"
    if not keep_punc:
        s = s.translate(str.maketrans('', '', string.punctuation))
    s = s.lower()  # downcase
    return s.split()


def process(strings):
    """
    Take in list of comments (strings) and tokenize them to build word index
    mappings to allowing conversion of comments into word vectors. Outputs are
    sequences of indices (used to map words to embeddings), the word->index
    and index->word) mappings and the frequencies of each word. The
    frequencies can be used to trim down the vocabulary to the most common
    words if the original size is too great.
    """
    word_index_map = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
    current_index = 3  # 0 is reserved for padding
    sequences = []
    index_word_map = ['<pad>', '<sos>', '<eos>']
    freqs = {'<sos>': 0, '<eos>': 0}
    for s in strings:
        sequence = []
        tokens = tokenizer(s, keep_punc=True)
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


def pad_seqs(sequences, padNum=0, front_pad=False):
    """
    Take in list of lists (sequences of variable length), and pad them with
    a given number (default: 0) to the same length (longest in sequences).
    """
    max_len = 0
    for i, seq in enumerate(sequences):
        max_len = len(seq) if len(seq) > max_len else max_len

    if not front_pad:
        sequences = [seq + [padNum]*(max_len - len(seq)) for seq in sequences]
    else:
        sequences = [[padNum]*(max_len - len(seq)) + seq for seq in sequences]
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
    inputs, targets, forced_inputs = load_samples(datapath, 10000)

    inputs, input_w2i, input_i2w, _ = process(inputs)
    input_words = [w for w in sorted(input_w2i, key=input_w2i.get)]
    targets, target_w2i, target_i2w, _ = process(targets)
    forced_inputs, forced_w2i, forced_i2w, _ = process(forced_inputs)

    # process sequence data for the network
    input_V, target_V = len(input_i2w), len(target_i2w)
    print('Input vocabulary size:', input_V)
    print('Target vocabulary size:', target_V)
    # pad sequences to same length and convert to numpy matrices
    inputs = np.array(pad_seqs(inputs, front_pad=True))
    targets = np.array(pad_seqs(targets))
    forced_inputs = np.array(pad_seqs(forced_inputs))

    print('inputs shape:', inputs.shape)
    print('targets shape:', targets.shape)
    print('forced_inputs shape:', forced_inputs.shape)

    # load and process pre-trained embeddings
    # http://nlp.stanford.edu/data/glove.6B.zip
    if os.path.isfile(datapath+'glove_embeddings/glove100_toxic.npy'):
        embeds = np.load(datapath+'glove_embeddings/glove100_toxic.npy')
    else:
        embeds = get_embeddings(
            datapath+'glove_embeddings/glove.6B.100d.txt',
            [i for i in range(len(input_words))], input_words
        )
        np.save(datapath+'glove_embeddings/glove100_toxic.npy', embeds)
    print('Word Embeddings shape:', embeds.shape)

    rnn = Seq2Seq(
        inputs.shape[1], targets.shape[1],  # maximum sequence lengths
        input_V, target_V,  # input and target vocabulary sizes
        256,  # LSTM latent dimension size
        1,  # number of stacked LSTM layers
        embeddings=embeds,
        LSTM_drop=0
    )
    rnn.fit(
        inputs, targets, forced_inputs, target_w2i, target_i2w,
        lr=1e-2, epochs=40, batch_sz=200
    )
    rnn.demo(inputs, input_i2w)


if __name__ == '__main__':
    datapath = 'C:/Users/geoff/GitRepos/learning/large_files/'
    main()
