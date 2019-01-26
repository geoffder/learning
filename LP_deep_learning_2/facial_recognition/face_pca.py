import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


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


def getData():
    print('loading in data...')
    df = pd.read_csv('fer2013.csv')
    print('data loaded.')
    N = df['emotion'].values.size

    pixels = np.array(
        [str.split(' ') for str in df['pixels'].values[:N//2]]
    ).astype(np.uint8)
    pixels = np.concatenate(
        [
            pixels,
            np.array(
                [str.split(' ') for str in df['pixels'].values[N//2:]]
            ).astype(np.uint8)
        ], axis=0
    )

    return pixels, df['emotion'].values


def plotExplained(pca):
    expl = pca.explained_variance_ratio_
    plt.plot(np.cumsum(expl))
    plt.show()


if __name__ == '__main__':
    X, T = getData()
    Xrebal, _ = classRebalance(X, T)
    # D = X.shape[1]
    Xnorm = (Xrebal - Xrebal.mean()) / Xrebal.std()
    pca = PCA(n_components=300)
    pca.fit(Xnorm)

    # plotExplained(pca)

    pixels = pd.DataFrame(pca.transform(X))
    emotion = pd.DataFrame(T)
    pixels.to_csv('pixels_300compPCA.csv', index=False)
    emotion.to_csv('emotion.csv', index=False)
