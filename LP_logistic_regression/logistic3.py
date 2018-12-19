import numpy as np
import matplotlib.pyplot as plt

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

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def crossEntropy(T, Y):
    return -(T * np.log(Y) + (1 - T)*np.log(1 - Y)).sum()

def costDeriv(X, Y, T):
    return (X.T @ (Y - T))

# find w with gradient descent
learning_rate = .01
tries = 100
w = np.random.randn(D+1) / np.sqrt(D+1) #bias col not included in D in this example
Y = sigmoid(Xb @ w) # try to predict Y with random weights
print('cross-entropy (random weights):', crossEntropy(T, Y))

for i in range(tries):
    w = w - learning_rate*costDeriv(Xb, sigmoid(Xb @ w), T)

# results
Y = sigmoid(Xb @ w)

print('w:', w)
# note that this is turning out a NaN due to a divide by zero when there is no error
# I guess this is why LP does it the way he does in logistic2.py. I guess I'll swtich
# to using that for simplicity.
# actually, on second thought, it is probably because of log(0)'s
# these may result in this set because of how easy it is to fit. Y is able to reach ~0,~1 (???)
# in the end, it is because the learning rate was too high, was going below 0 instead of approaching
# leaving this here as a reminder to not be an idiot.
print('cross-entropy:', crossEntropy(T, Y))

# plot results
x_axis = np.linspace(-6,6,N)
y_axis = -x_axis
# plot points coloured by value of T
plt.scatter(X[:,0],X[:,1], c=T, s=100, alpha=.5)
# plot points coloured by predicted value Y on top (transperencies will stack)
plt.scatter(X[:,0],X[:,1], c=(np.round(Y)), s=100, alpha=.5)
# looks like the ones on the line disappear when these opposite colours are stacked.
plt.plot(x_axis,y_axis)
plt.plot(x_axis, Y) # range 0 to 1. (prediction of which class, notice how it crosses the axis)
plt.show()
