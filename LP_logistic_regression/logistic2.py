import numpy as np

N = 100
D = 2

X = np.random.randn(N, D)

# makeing the two classes (each row of X is an (x,y) point)
# centre the two 2d gaussians on seperate points, so they are two seperable pops
X[:50,:] = X[:50,:] - 2*np.ones((50,D)) # centred at X=-2 and Y=-2
X[50:, :] = X[50:,:] + 2*np.ones((50,D)) # centred at X=2 and Y=2

T = np.array([0]*50 + [1]*50)
ones = np.ones((N,1))
Xb = np.concatenate((ones, X), axis=1) # add column of ones (bias term)

# randomly initialize weights
w = np.random.randn(D + 1)

# calculate the model output
z = Xb @ w

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

Y = sigmoid(z)

# my version
def crossEntropy(T, Y):
    return -(T * np.log(Y) + (1 - T)*np.log(1 - Y)).sum()

# LPs version
def crossEntropy_LP(T, Y):
    E = 0
    for i in range(N):
        if T[i] == 1:
            E -= np.log(Y[i]).sum()
        else:
            E -= np.log(1 - Y[i]).sum()
    return E

cost = crossEntropy(T, Y)
cost_LP = crossEntropy_LP(T, Y)

print('random weights cost:', cost)
print('random weights cost by LP:', cost_LP)

# we can use the closed form solution to calculate w here because we have
# two normal dists with equal variance

# the numbers in the notes. Check out by hand to see how these are arrived at
w = np.array([0, 4, 4])
z = Xb @ w
Y = sigmoid(z)

cost = crossEntropy(T, Y)
print('closed-form weights cost:', cost)
