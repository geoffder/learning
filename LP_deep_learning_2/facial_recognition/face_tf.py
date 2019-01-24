import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle


def classRebalance(X, T):
    classes = np.unique(T)
    Xlist = [X[T == k] for k in classes]
    Tlist = [T[T == k] for k in classes]
    bigN = np.max([t.shape for t in Tlist])
    # develop a less coarse way that better approximates the classes
    Xlist = [np.concatenate([x]*(bigN//x.shape[0]), axis=0) for x in Xlist]
    Tlist = [np.concatenate([t]*(bigN//t.shape[0]), axis=0) for t in Tlist]

    return np.concatenate(Xlist, axis=0), np.concatenate(Tlist, axis=0)


if __name__ == '__main__':
    print('loading in data...')
    df = pd.read_csv('fer2013.csv')
    print('data loaded.')
    print('samples in full dataset:', df['emotion'].values.size)
    maxN = 10000
    # pixels = []
    # for str in df['pixels'].values[:maxN]:
    #     pixels.append(str.split(' '))
    # pixels = np.array(pixels).astype(np.uint8)
    pixels = np.array(
        [str.split(' ') for str in df['pixels'].values[:maxN]]
    ).astype(np.uint8)
    print(pixels.shape)
    X, T = classRebalance(pixels, df['emotion'].values[:maxN])
    print('X shape:', X.shape, 'T shape:', T.shape)
    print('emotion counts:', [(T == k).sum() for k in np.unique(T)])
