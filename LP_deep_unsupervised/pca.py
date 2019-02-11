import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from LP_util import getKaggleMNIST


def main():
    Xtrain, Ttrain, Xtest, Ttest = getKaggleMNIST()

    pca = PCA()
    reduced = pca.fit_transform(Xtrain)
    # plot the first two dimensions of the transformed data
    plt.scatter(reduced[:, 0], reduced[:, 1], s=100, c=Ttrain, alpha=.5)
    plt.show()

    # cumulative, last = [], 0
    # for var in pca.explained_variance_ratio_:
    #     cumulative.append(var + last)
    #     last = cumulative[-1]
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    fig, ax = plt.subplots(1, 1)
    ax.plot(cumulative)
    ax.set_xlabel('dimensions')
    ax.set_ylabel('variance explained')
    plt.show()


if __name__ == '__main__':
    main()
