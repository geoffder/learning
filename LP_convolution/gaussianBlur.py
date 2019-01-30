import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
import matplotlib.image as mpimg

# load and show original image
img = mpimg.imread('lena.png')
# plt.imshow(img)
# plt.show()

# first make the image black and white since the convolve2d function only
# works on 2D matrices.(not RGB)
bw = img.mean(axis=2)
# plt.imshow(bw, cmap='gray')
# plt.show()


def createFilter(k):
    'Create a gaussian blur filter'
    W = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            dist = (i - 9.5)**2 + (j - 9.5)**2  # distance from centre
            W[i, j] = np.exp(-dist / 50)
    return W


W = createFilter(20)
# plt.imshow(W, cmap='gray')
# plt.show()

# apply filter to greyscale lena img
blurred = convolve2d(bw, W)
# plt.imshow(blurred, cmap='gray')
# plt.show()

# check the shape. See that the output is larger than the input.
# full-padding was done by the convolution function
print('in shape:', bw.shape, '; out shape:', blurred.shape)
# can pass params to convolution2d to make them the same
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html

# this crops out the edges that extend past the original size.
# padded region dropped after convolution.
blurred = convolve2d(bw, W, mode='same')
# plt.imshow(blurred, cmap='gray')
# plt.show()
print("convole2d with mode='same'")
print('in shape:', bw.shape, '; out shape:', blurred.shape)

rgbBlur = np.dstack([convolve2d(img[:, :, chan], W, mode='same')
                    for chan in range(img.shape[2])])
rgbBlur /= rgbBlur.max()  # put in range 0->1, valid range for float images
# other option is to normalize the filter BEFORE convolving
# e.g. W /= W.sum()

plt.imshow(rgbBlur)
plt.show()
