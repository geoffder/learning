import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# make a donut
def getXOR(N):
    quads = [[.5, .5], [-.5, -.5], [.5, -.5], [-.5, .5]]
    X = np.concatenate(
        [np.random.random((N//4, 2)) + quad for quad in quads],
        axis=0
    )
    Y = [0]*(N//2) + [1]*(N//2)
    return X, Y


def main():
    N = 600
    X, labels = getXOR(N)

    # display XOR problem
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.show()

    # embed with t-SNE
    X_embedded = TSNE(n_components=2, perplexity=40).fit_transform(X)
    print('shape of embedded X (2 components):', X_embedded.shape)
    # plot the 2D embedding of the 2D data
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, alpha=.3)
    plt.show()

    '''
    Unsurprisingly, t-SNE does not really do much, as it acts on the distances
    between points, and in this example of XOR, all the points are uniformly
    distributed in one big block. There is nothing to differentiate the points
    for this algorithm.
    '''


if __name__ == '__main__':
    main()
