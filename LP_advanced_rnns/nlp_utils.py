import numpy as np
import pandas as pd

from sklearn.utils import shuffle
import string

"""
Collection of simple utilities I've made for use on NLP tasks, particularly
the Advanced NLP and RNNs course by Lazy Programmer.
"""


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
    """
    Load pre-trained embeddings and build a matrix including only the word
    vectors for those indicated in the common lists. Allows trimming down
    the enormous embedding files down for topic specific tasks.
    """
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


def trainTestSplit_3way(X, T, F, ratio=.5):
    '''
    Shuffle dataset and split into training and validation sets given a
    train:test ratio. Make a general one that can handle a flexible number of
    arrays soon.
    '''
    X, T, F = shuffle(X, T, F)
    N = X.shape[0]
    Xtrain, Ttrain = X[:int(N*ratio)], T[:int(N*ratio)]
    Ftrain = F[:int(N*ratio)]
    Xtest, Ttest = X[int(N*ratio):], T[int(N*ratio):]
    Ftest = F[int(N*ratio):]
    return Xtrain, Ttrain, Ftrain, Xtest, Ttest, Ftest
