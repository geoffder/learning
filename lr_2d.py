import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# load the data
X = []
Y = []
for line in open('data_2d.csv'):
    x1,x2,y = line.split(',')
    X.append([float(x1), float(x2), 1]) # adding in the x0 bias term
    Y.append(float(y))

# convert to numpy array objects
X = np.array(X)
Y = np.array(Y)

# calculate w (fit parameter vector)
w = np.linalg.solve(X.T @ X, X.T @ Y) # using matrix multiply operator @
# w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y)) # cleaner transpose syntax
#w = np.linalg.solve(np.transpose(X).dot(X), np.transpose(X).dot(Y))
# all of the above solutions are identical, cleanest one used.

# calculate fit
Yhat = X @ w # @ is the matrix multiplication operator

# calculate r-squared
SSres = ((Y - Yhat)**2).sum()
SStot = ((Y - Y.mean())**2).sum()
Rsq = 1.0 - SSres/SStot
print("r-squared: " + str(Rsq))

#plot the data
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(X[:,0], X[:,1],Y) # all rows col0, all rows col1, Y
ax.scatter(X[:,0], X[:,1],Yhat)
plt.show()
