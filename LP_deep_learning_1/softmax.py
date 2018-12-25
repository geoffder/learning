import numpy as np

# our input will be the activation (a) values at the last layer
a = np.random.randn(5) # five output classes/units

# each prediction is exp of activation divided by exp of all other activations
def softmax_1sample(a):
    return np.exp(a) / np.exp(a).sum()

print(softmax_1sample(a))
print('softmax sum:', softmax_1sample(a).sum()) # should be equal to 1

# now lets do it for multiple samples (A is a matrix of NxK)
N = 100
K = 5
A = np.random.randn(N, K)

# must sum along axis one, so it is the sum of exps for each sample seperately
# keep dims prevents us from getting dimesion missmatch (without, we would have
# (100,5) / (100,). With you get (100,5) / (100,1) and np broadcasts for us.
# problem is 2D vs 1D array, with keepdims the sum keeps it as 2D
def softmax(A):
    return np.exp(A) / np.sum(np.exp(A), axis=1, keepdims=True)

#print(np.sum(np.exp(A), axis=1, keepdims=True).shape)
#print(np.sum(np.exp(A), axis=1).shape)
# sum across columns within each sample to show they are probabilities
print(np.sum(softmax(A),axis=1))
