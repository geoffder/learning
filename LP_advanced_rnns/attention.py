import numpy as np
import matplotlib.pyplot as plt

import os.path
from nlp_utils import load_samples, pad_seqs, process, get_embeddings
from nlp_utils import trainTestSplit_3way

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim


# use GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


class Contextualizer(nn.Module):
    """

    Note: Assumes batch_first LSTMs are being used.
    """
    def __init__(self, encode_dim_sz, decode_dim_sz):
        super(Contextualizer, self).__init__()
        self.encode_dim_sz = encode_dim_sz  # encoder latent dimensions
        self.decode_dim_sz = decode_dim_sz  # decoder latent dimensions
        self.build()

    def build(self):
        # reducing down to encoder dims here, could make it a variable though
        self.layer1 = nn.Linear(
            self.decode_dim_sz+self.encode_dim_sz*2, self.encode_dim_sz*2
        )
        # one alpha weight per timestep
        self.layer2 = nn.Linear(self.encode_dim_sz*2, 1)

    def forward(self, state, code):
        state = state.transpose(0, 1)  # from (TxNxD) to (NxTxD)
        Z = torch.cat([state.expand(-1, code.shape[1], -1), code], dim=2)
        alpha = F.softmax(self.layer2(torch.tanh(self.layer1(Z))), dim=1)

        # shapes: alpha (NxTx1), code (NxTxD). matmul does 2d mul with batches
        context = torch.matmul(alpha.transpose(1, 2), code)
        return context


class Attention(nn.Module):
    """
    Sequence-to-sequence machine translation model, with Attention.
    """
    def __init__(self, in_len, targ_len, in_V, targ_V, encode_dim_sz,
                 decode_dim_sz, num_LSTM_layers, teacher_forcing=True,
                 blend_context=False, embeddings=None, LSTM_drop=0):
        super(Attention, self).__init__()
        # data shape
        self.in_len = in_len
        self.targ_len = targ_len
        self.in_V = in_V  # input vocabulary size
        self.targ_V = targ_V  # target vocabulary size
        # architecture
        self.encode_dim_sz = encode_dim_sz  # encoder latent dimensions
        self.decode_dim_sz = decode_dim_sz  # decoder latent dimensions
        self.num_LSTM_layers = num_LSTM_layers
        self.teacher_forcing = teacher_forcing
        self.blend_context = blend_context  # mix last vector with context
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
            # stick with 100 embedding dimensions for simplicity
            self.encoder_embed = nn.Embedding(self.in_V, 100, padding_idx=0)

        # attention block, takes last decoder state and encoder output
        # returns context vector for input into decoder
        self.context_block = Contextualizer(
            self.encode_dim_sz, self.decode_dim_sz
        )

        # encoder and decoder recurrent networks
        self.encoder_LSTM = nn.LSTM(
            100, self.encode_dim_sz, self.num_LSTM_layers,
            bias=True, bidirectional=True, dropout=self.LSTM_drop,
            batch_first=True
        )
        # teacher forcing concats embedding dim onto context vector. If
        # blend_context is used, dimensionality is reduced back down.
        in_sz = self.encode_dim_sz*2
        in_sz += 100 if self.teacher_forcing and not self.blend_context else 0
        self.decoder_LSTM = nn.LSTM(
            in_sz, self.decode_dim_sz, self.num_LSTM_layers,
            bias=True, dropout=self.LSTM_drop, batch_first=True
        )

        # teacher forcing layers (decoder embeddings and dim reduction)
        if self.teacher_forcing:
            # decoder network word embeddings (randomly initialized)
            self.decoder_embed = nn.Embedding(self.targ_V, 100, padding_idx=0)
            # dimensionality reduction layer (keep at self.encode_dim_sz*2)
            if self.blend_context:
                self.dim_reducer = nn.Linear(
                    self.encode_dim_sz*2 + 100, self.encode_dim_sz*2
                )

        # fully connected layers for decoder -> word probabilities
        self.logistic = nn.Linear(self.decode_dim_sz, self.targ_V)

    def teaching(self, X, T):
        """
        Takes both input and target (offset by <sos> token) sequences, as the
        offset targets are fed in to the decoder network, rather than the
        decoder's own predictions (as they would be during generation). This
        ensures that the model is able to learn the entire sequence during
        every pass, rather than de-railing and training on garbage after one
        false prediction.

        Note: If teacher_forcing is not activated, the target sequences are
        never used. Without it, the input to the decoder LSTM is just the
        context vector (attention).
        """
        # convert sequences of indices to word-vector matrices
        X = self.encoder_embed(X)  # shape: (batch, T, D)
        if self.teacher_forcing:
            T = self.decoder_embed(T)
        # get bi-directional encoding of input sequence
        encoding, _ = self.encoder_LSTM(X)  # shape: (batch, T, D*2)

        # loop over time and generate output with decoder using attention
        output = []
        s = torch.zeros([1, X.shape[0], self.decode_dim_sz]).float().to(device)
        c = torch.zeros([1, X.shape[0], self.decode_dim_sz]).float().to(device)
        for t in range(self.targ_len):
            # calculate context with with s(t-1) and encoder output
            context = self.context_block(s, encoding)
            if self.teacher_forcing:
                forced = T[:, t, :].unsqueeze(1)  # vector for correct output
                # tack previous (forced) output on to context vector
                context = torch.cat([context, forced], dim=2)
                if self.blend_context:
                    context = torch.tanh(self.dim_reducer(context))

            # get next decoder output (and update states)
            _, (s, c) = self.decoder_LSTM(context, (s, c))
            output.append(s.squeeze())  # remove time dimension

        # get logits
        output = torch.stack(output, dim=1)  # stack on time dimension
        return self.logistic(output)

    def translate(self, X):
        """
        Forward pass without teacher forcing. Model will predict the next word
        of the output based on the translation SO FAR, regardless of whether
        it is correct or not. Used for testing during fitting, and also for
        translating sequences after training is over.

        Note: only adds previous output embedding to context if
        teacher_forcing is being used. Quite slow. Try making a batched version
        as an exercise and see how much faster I can get it. 
        """
        out = np.zeros((X.shape[0], self.targ_len))
        for i in range(X.shape[0]):
            s = torch.zeros(
                [1, 1, self.decode_dim_sz]).float().to(device)
            c = torch.zeros(
                [1, 1, self.decode_dim_sz]).float().to(device)
            in_embed = self.encoder_embed(X[i].view(1, -1))
            encoding, _ = self.encoder_LSTM(in_embed)
            seq = [self.targ_w2i['<sos>']]
            for t in range(self.targ_len):
                # calculate context with with s(t-1) and encoder output
                context = self.context_block(s, encoding)
                if self.teacher_forcing:
                    vec = self.decoder_embed(
                        torch.LongTensor([[seq[-1]]]).to(device)
                    )
                    context = torch.cat([context, vec], dim=2)
                    # blend previous output in with context vector
                    if self.blend_context:
                        context = torch.tanh(self.dim_reducer(context))

                # get next decoder output (and update states)
                _, (s, c) = self.decoder_LSTM(context, (s, c))
                pred = torch.argmax(self.logistic(s[-1])).long()
                seq.append(pred.detach().cpu().tolist())
                if seq[-1] == self.targ_w2i['<eos>']:
                    break

            seq = seq[1:] + [self.targ_w2i['<pad>']]*(self.targ_len-t-1)
            out[i, :] = np.array(seq)  # ignore <sos> token
        return torch.from_numpy(out).long().to(device)

    def fit(self, Xtrain, Ttrain, Ftrain, Xtest, Ttest, Ftest, targ_w2i,
            targ_i2w, lr=1e-4, epochs=40, batch_sz=200, print_every=50):

            N = Xtrain.shape[0]  # number of samples
            self.targ_w2i = targ_w2i  # word to index for target vocabulary
            self.targ_i2w = targ_i2w  # index to word for target vocabulary

            # send data to GPU
            Xtrain = torch.from_numpy(Xtrain).long().to(device)  # inputs
            Xtest = torch.from_numpy(Xtest).long().to(device)
            Ttrain = torch.from_numpy(Ttrain).long().to(device)  # targets
            Ttest = torch.from_numpy(Ttest).long().to(device)
            Ftrain = torch.from_numpy(Ftrain).long().to(device)  # forced
            Ftest = torch.from_numpy(Ftest).long().to(device)

            self.loss = nn.CrossEntropyLoss(
                reduction='mean', ignore_index=0).to(device)
            # self.loss = nn.CrossEntropyLoss(reduction='mean').to(device)
            self.optimizer = optim.Adam(self.parameters(), lr=lr)

            n_batches = N // batch_sz
            train_costs, train_accs, test_accs = [], [], []
            for i in range(epochs):
                cost = 0
                print("epoch:", i, "n_batches:", n_batches)
                # shuffle dataset for next epoch of batches
                inds = torch.randperm(Xtrain.size()[0])
                Xtrain, Ttrain = Xtrain[inds], Ttrain[inds]
                Ftrain = Ftrain[inds]
                for j in range(n_batches):
                    Xbatch = Xtrain[j*batch_sz:(j*batch_sz+batch_sz)]
                    Tbatch = Ttrain[j*batch_sz:(j*batch_sz+batch_sz)]
                    Fbatch = Ftrain[j*batch_sz:(j*batch_sz+batch_sz)]

                    cost += self.train_step(Xbatch, Tbatch, Fbatch)

                    if j % print_every == 0:
                        # shuffle test set
                        inds = torch.randperm(Xtest.size()[0])
                        Xtest, Ttest = Xtest[inds], Ttest[inds]
                        Ftest = Ftest[inds]
                        # validation cost and accuracy with forced teaching
                        forced_cost, forced_acc = self.teaching_score(
                            Xtest, Ttest, Ftest
                        )
                        print(
                            'teaching cost: %.2f, teaching acc: %.3f'
                            % (forced_cost, forced_acc)
                        )
                        # accuracy during 'free' translation (non-forced)
                        # only sample, since it is much slower than forced
                        trans_acc = self.trans_score(
                            Xtest[:batch_sz], Ttest[:batch_sz]
                        )
                        print("translation accuracy: %.3f" % (trans_acc))

                # for plotting
                train_costs.append(cost / n_batches)
                train_accs.append(forced_acc)
                test_accs.append(trans_acc)

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
        logits = self.teaching(inputs, forced_inputs)
        output = self.loss.forward(logits.transpose(2, 1), targets)

        # Backward
        output.backward()
        self.optimizer.step()  # Update parameters

        return output.item()

    def teaching_score(self, inputs, targets, forced_inputs, pad_corr=False):
        """
        Get cost and accuracy of a forced_teaching run. Measure of how well
        model is able to predict the next word of the translated sequence,
        using the correct sequence as the input, regardless of the actual
        prediction.
        """
        # forward pass using forced teaching
        self.eval()  # set the model to testing mode
        with torch.no_grad():
            logits = self.teaching(inputs, forced_inputs)
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
        self.eval()
        inputs = torch.from_numpy(sequences).long().to(device)
        while True:
            idx = np.random.randint(0, sequences.shape[0])
            with torch.no_grad():
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

    # train-test split
    Xtrain, Ttrain, Ftrain, Xtest, Ttest, Ftest = trainTestSplit_3way(
        inputs, targets, forced_inputs, ratio=.8
    )

    # load and process pre-trained embeddings
    # http://nlp.stanford.edu/data/glove.6B.zip
    if os.path.isfile(datapath+'glove_embeddings/glove100_trans.npy'):
        embeds = np.load(datapath+'glove_embeddings/glove100_trans.npy')
    else:
        embeds = get_embeddings(
            datapath+'glove_embeddings/glove.6B.100d.txt',
            [i for i in range(len(input_words))], input_words
        )
        np.save(datapath+'glove_embeddings/glove100_trans.npy', embeds)
    print('Word Embeddings shape:', embeds.shape)

    rnn = Attention(
        inputs.shape[1], targets.shape[1],  # maximum sequence lengths
        input_V, target_V,  # input and target vocabulary sizes
        256,  # encoder LSTM latent dimension size
        256,  # decoder LSTM latent dimension size
        1,  # number of stacked LSTM layers (only 1 works atm)
        teacher_forcing=True,
        blend_context=True,
        embeddings=embeds,
        LSTM_drop=0
    )
    rnn.fit(
        Xtrain, Ttrain, Ftrain, Xtest, Ttest, Ftest, target_w2i, target_i2w,
        lr=1e-3, epochs=40, batch_sz=100, print_every=50
    )
    rnn.demo(Xtest, input_i2w)


if __name__ == '__main__':
    datapath = 'C:/Users/geoff/GitRepos/learning/large_files/'
    main()
