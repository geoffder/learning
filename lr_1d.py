import numpy as np
import matplotlib.pyplot as plt

# load the 1d data provided (2 cols, comma delimited)
X = []
Y = []
for line in open('data_1d.csv'):
    x, y = line.split(',')
    X.append(float(x))
    Y.append(float(y))

# convert to numpy arrays
X = np.array(X)
Y = np.array(Y)

# calculate a and b using the equations learned in prev section
denom = X.dot(X) - X.mean() * X.sum()
a = (Y.dot(X) - X.mean() * Y.sum())/denom
b = (Y.mean()*X.dot(X) - X.mean()*Y.dot(X))/denom
print("a: " + str(a))
print("b: " + str(b))

# predict y with a and b
Yhat = np.add(np.multiply(X,a),b)
# or simply Yhat = a*X + b since they are already numpy array objects

# plot
plt.scatter(X, Y)s
plt.plot(X,Yhat)
plt.show()
