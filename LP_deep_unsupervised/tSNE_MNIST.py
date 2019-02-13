import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.manifold import TSNE
from LP_util import getKaggleMNIST


def main():
    Xtrain, Ttrain, Xtest, Ttest = getKaggleMNIST()
    N = 1000

    if 0:
        fig, axes = plt.subplots(10, 10, figsize=(15, 15))
        for i, row in enumerate(axes):
            for j, col in enumerate(row):
                col.imshow(Xtrain[i*axes.shape[0]+j].reshape(28, 28),
                           cmap='gray')
        plt.show()

    # embed with t-SNE into 2 dimensions
    X_embedded = TSNE(
        n_components=2, perplexity=30).fit_transform(Xtrain[:N])
    print('shape of embedded X (2 components):', X_embedded.shape)
    # plot the 2D embedding of the 2D data
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=Ttrain[:N],
                alpha=.5, s=100)
    plt.show()

    # embed with t-SNE into 3 dimensions
    X_embedded = TSNE(
        n_components=3, perplexity=30).fit_transform(Xtrain[:N])
    print('shape of embedded X (3 components):', X_embedded.shape)
    # plot the 2D embedding of the 2D data
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.scatter(
        X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2],
        c=Ttrain[:N], alpha=.5, s=100
    )
    plt.show()


if __name__ == '__main__':
    main()
