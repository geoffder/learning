import numpy as np
import matplotlib.pyplot as plt

N = 100
D = 2

X = np.random.randn(N, D)

# makeing the two classes (each row of X is an (x,y) point)
# centre the two 2d gaussians on seperate points, so they are two seperable pops
X[:50,:] = X[:50,:] - 2*np.ones((50,D)) # centred at X=-2 and Y=-2
X[50:, :] = X[50:,:] + 2*np.ones((50,D)) # centred at X=2 and Y=2

T = np.array([0]*50 + [1]*50) # first 50 pts are class 0, last 50 are class 1
# add a column of ones
ones = np.ones((N,1))
Xb = np.concatenate((ones, X), axis=1) # add column of ones (bias term, first column this time)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

# already know the closed form solution (see notes)
w = [0, 4 , 4] # bias (or b) = 0

Y = sigmoid(Xb @ w)
x_axis = np.linspace(-6,6, N) # to plot Y against (same number of points)
y_axis = -x_axis
# set colour of markers based on value of T
plt.scatter(X[:,0], X[:,1], c=T, s=100, alpha=.5) #s=size, alpha=transparency
plt.plot(x_axis, y_axis)
plt.plot(x_axis, Y) # see how Y ~ 0 on left of dividing line, and Y ~ 1 on right
plt.show()
