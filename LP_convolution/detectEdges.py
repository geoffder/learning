import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
import matplotlib.image as mpimg

img = mpimg.imread('lena.png')
bw = img.mean(axis=2)

# Sobel operator
# filters = [np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).astype(np.float32),
#            np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.float32)]

# filters = [np.array([[1, 0, -1]]), np.array([[-1, 0, 1]]),
#            np.array([[1, 0, -1]]).T, np.array([[-1, 0, 1]]).T]

# filters = [np.array([[1, -2, 1]]), np.array([[1, -2, 1]]).T,
#            np.array([[-.25, 0, .25], [0, 0, 0], [.25, 0, -.25]])]

# edges = bw[:]
# for f in filters:
#     edges = convolve2d(edges, f, mode='same')

# plt.imshow(edges, cmap='gray')
# plt.show()

# horizontal and vertical edge filters
Hx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).astype(np.float32)
Hy = Hx.T

# think of these as vectors that we can calc the magnitude and direction of
Gx = convolve2d(bw, Hx, mode='same')
Gy = convolve2d(bw, Hy, mode='same')

# gradients magnitude
G = np.sqrt(Gx**2 + Gy**2)
plt.imshow(G, cmap='gray')
plt.show()

# gradients direction
theta = np.arctan2(Gy, Gx)
plt.imshow(theta, cmap='gray')
plt.show()
