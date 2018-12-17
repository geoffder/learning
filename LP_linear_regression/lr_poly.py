import numpy as np
import matplotlib.pyplot as plt

# load the data
X = []
Y = []
for line in open('data_poly.csv'):
    x,y = line.split(',')
    x = float(x)
    X.append([1, x, x**2]) # since polynomial, setting up differently
    Y.append(float(y))

# convert to numpy arrays
X = np.array(X)
Y = np.array(Y)

# peak at the data
# plt.scatter(X[:,1],Y) # plotting the x^2 vector instead linearizes.
# plt.show()

# calculate weights for fitting (weighing betw x and x^2. Will be mostly x^2)
w = np.linalg.solve(X.T @ X, X.T @ Y)
print("w (x^2, x, bias [I think?]): " + str(w))
# make predictions
Yhat = X @ w
# r-squared of fit
SSres = ((Y - Yhat)**2).sum()
SStot = ((Y - Y.mean())**2).sum()
Rsq = 1.0 - SSres/SStot
print("r-squared: " + str(Rsq))

# plot data with predictions
plt.scatter(X[:,1], Y)
#plt.scatter(X[:,1], Yhat)
plt.plot(sorted(X[:,1]), sorted(Yhat)) # function is monotonically increasing, so this is ok
plt.show()
