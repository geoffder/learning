import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datetime import datetime

'Implementations of 2D convolution without the use of a library (besides np)'


def convolve2dSlow(X, W):
    'Slowest implementation, loop over all elements individually'
    t0 = datetime.now()
    n1, n2 = X.shape
    m1, m2 = W.shape
    Y = np.zeros((n1 + m1 - 1, n2 + m2 - 1))
    for i in range(n1 + m1 - 1):
        for ii in range(m1):
            for j in range(n2 + m2 - 1):
                for jj in range(m2):
                    if i >= ii and j >= jj and i - ii < n1 and j - jj < n2:
                        Y[i, j] += W[ii, jj]*X[i - ii, j - jj]
    print("elapsed time:", (datetime.now() - t0))
    return Y


def convolve2d(X, W):
    'Use element-wise matrix operations of numpy arrays instead of loops'
    t0 = datetime.now()
    n1, n2 = X.shape
    m1, m2 = W.shape
    Y = np.zeros((n1 + m1 - 1, n2 + m2 - 1))
    for i in range(n1):
        for j in range(n2):
            Y[i:i+m1, j:j+m2] += X[i, j]*W
    print("elapsed time:", (datetime.now() - t0))
    return Y


def convolve2dSame(X, W):
    'Output same size as input, crop off padding'
    n1, n2 = X.shape
    m1, m2 = W.shape
    Y = np.zeros((n1 + m1 - 1, n2 + m2 - 1))  # full pad for calculations
    for i in range(n1):
        for j in range(n2):
            Y[i:i+m1, j:j+m2] += X[i, j]*W
    ret = Y[m1//2:-m1//2+1, m2//2:-m2//2+1]
    assert(ret.shape == X.shape)
    return ret


# smaller than input
def convolve2dSmall(X, W):
    'Output smaller than input, not using edges influenced by padding'
    n1, n2 = X.shape
    m1, m2 = W.shape
    Y = np.zeros((n1 + m1 - 1, n2 + m2 - 1))  # full pad for calculations
    for i in range(n1):
        for j in range(n2):
            Y[i:i+m1, j:j+m2] += X[i, j]*W
    ret = Y[m1-1:-m1+1, m2-1:-m2+1]
    return ret


# load the famous Lena image
img = mpimg.imread('lena.png')

# what does it look like?
plt.imshow(img)
plt.show()

# make it B&W
bw = img.mean(axis=2)
plt.imshow(bw, cmap='gray')
plt.show()

# create a Gaussian filter
W = np.zeros((20, 20))
for i in range(20):
    for j in range(20):
        dist = (i - 9.5)**2 + (j - 9.5)**2
        W[i, j] = np.exp(-dist / 50.)

# let's see what the filter looks like
plt.imshow(W, cmap='gray')
plt.show()

# now the convolution
out = convolve2dSame(bw, W)
plt.imshow(out, cmap='gray')
plt.show()

# what's that weird black stuff on the edges? let's check the size of output
print(out.shape)
# after convolution, the output signal is N1 + N2 - 1

# try it in color
out = np.zeros(img.shape)
W /= W.sum()
for i in range(3):
    out[:, :, i] = convolve2dSame(img[:, :, i], W)
plt.imshow(out)
plt.show()
