import numpy as np
import matplotlib.pyplot as plt
from LP_util import getKaggleMNIST

Xtrain, Ttrain, Xtest, Ttest = getKaggleMNIST()

# decompose covariance
covX = np.cov(Xtrain.T)  # covariance matrix
lambdas, Q = np.linalg.eigh(covX)  # get eigenvalues and eigenvectors
# lambdas represent how much variance each vector Q accounts for

# sort by descending information
idx = np.argsort(-lambdas)  # negative so descending rather than ascending
lambdas = np.maximum(lambdas[idx], 0)  # get rid of negatives
Q = Q[:, idx]

# plot first two columns of Z
# transform X into Z by multiplying it with the sorted Q matrix
Z = Xtrain.dot(Q)
plt.scatter(Z[:, 0], Z[:, 1], s=100, c=Ttrain, alpha=.3)
plt.show()

cumulative = np.cumsum(lambdas / lambdas.sum())  # div by sum so adds to 1
fig, ax = plt.subplots(1, 1)
ax.plot(cumulative)
ax.set_xlabel('dimensions')
ax.set_ylabel('cumulative explained variance')
plt.show()
