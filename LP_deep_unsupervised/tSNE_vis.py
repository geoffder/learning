import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.manifold import TSNE

coords = [0, 5]
corners = np.array([[i, j, k] for i in coords for j in coords for k in coords])

# take a look at the cube defined by these corners in 3D
if 0:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(corners[:, 0], corners[:, 1], corners[:, 2])
    plt.show()

# gaussian clouds centred on the corners
X = np.concatenate(
    [np.random.randn(20, 3)+corner for corner in corners],
    axis=0
)
# different label for each cloud
labels = np.concatenate([[i]*20 for i in range(len(corners))])

# display coloured clouds centred on corners of cube
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels)
plt.show()

# perform t-SNE
X_embedded = TSNE(n_components=2).fit_transform(X)

# plot the 2D embedding of the 3D data
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels)
plt.show()
