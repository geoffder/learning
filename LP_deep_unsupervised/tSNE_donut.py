import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# make a donut
def getDonut(N, R_inner=5, R_outer=10):
    R1 = np.random.randn(N//2) + R_inner
    theta = 2 * np.pi * np.random.random(N//2)  # polar coordinates
    # use polar coordinates to create (x,y) coordinates
    X_inner = np.concatenate([[R1*np.cos(theta)], [R1*np.sin(theta)]]).T

    R2 = np.random.randn(N//2) + R_outer
    theta = 2 * np.pi * np.random.random(N//2)
    X_outer = np.concatenate([[R2*np.cos(theta)], [R2*np.sin(theta)]]).T

    X = np.concatenate([X_inner, X_outer], axis=0)
    Y = [0]*(N//2) + [1]*(N//2)
    return X, Y


def main():
    N = 600
    X, labels = getDonut(N, R_inner=10, R_outer=20)

    # display donut
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.show()

    # try to reduce to 1D as is using t-SNE
    X_embedded = TSNE(perplexity=40, n_components=1).fit_transform(X)
    print('shape of embedded X (1 component):', X_embedded.shape)
    # plot the 1D embedding of the 2D data
    plt.scatter(X_embedded[:, 0], X_embedded[:, 0], c=labels)
    plt.show()

    # if not reducing the dimensionality
    X_embedded = TSNE(n_components=2, perplexity=40).fit_transform(X)
    print('shape of embedded X (2 components):', X_embedded.shape)
    # plot the 2D embedding of the 2D data
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels)
    plt.show()


if __name__ == '__main__':
    main()
