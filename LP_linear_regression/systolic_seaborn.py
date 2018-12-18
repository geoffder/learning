# need to sudo pip install xlrd to use pd.read_excel
# data is from:
# http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/mlr/frames/mlr02.html

# The data (X1, X2, X3) are for each patient.
# X1 = systolic blood pressure (treat as Y)
# X2 = age in years (using to predict Y)
# X3 = weight in pounds (using to predict Y)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_excel('mlr02.xls')
df.rename(columns={'X1': 'BP', 'X2': 'Age', 'X3': 'Weight'}, inplace = True)
data = df.values

# take a look at the data using seaborn
f, axes = plt.subplots(1, 2)
f.tight_layout() # plt property that automatically spaces subplots
# plt.subplot_tool() # this command makes a manual adjustment panel appear with figure
sns.scatterplot(x='Age', y='BP', data = df, ax = axes[0])
#axes[0].set(xlabel='age (years)', ylabel='systolic blood pressure')
sns.scatterplot(x='Weight', y='BP', data = df, ax = axes[1])
#axes[1].set(xlabel='weight (lbs)', ylabel='systolic blood pressure')
plt.show()

# (trying looking in 3d for fun)
# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')
# ax.scatter(X[:,1], X[:,2],X[:,0])
# plt.show()

# pull out X and Y
X = []
Y = []
for line in data:
    X.append([float(line[1]), float(line[2]), 1.])
    Y.append(float(line[0]))
X = np.array(X)
Y = np.array(Y)

# years vs blood pressure (1d)
years = X[:,0]
denom = years.dot(years) - years.mean() * years.sum()
yr_a = (Y.dot(years) - years.mean() * Y.sum()) / denom
yr_b = (Y.mean()*years.dot(years) - years.mean()*Y.dot(years))/denom
yr_Yhat = yr_a*years + yr_b

yr_SSres = ((Y - yr_Yhat)**2).sum()
SStot = ((Y - Y.mean())**2).sum() # same for all fits
yr_Rsq = 1. - yr_SSres/SStot
print("years only Rsq: " + str(yr_Rsq))

f, axes = plt.subplots(1,1)
sns.scatterplot(x = years, y = Y, ax = axes)
sns.lineplot(x = years, y = yr_Yhat, ax = axes)
axes.set(xlabel='age (years)', ylabel='predicted sys blood pressure')
plt.show()

# weight vs blood pressure (1d)
weights = X[:,1]
denom = weights.dot(weights) - weights.mean() * weights.sum()
wt_a = (Y.dot(weights) - weights.mean() * Y.sum()) / denom
wt_b = (Y.mean()*weights.dot(weights) - weights.mean()*Y.dot(weights))/denom
wt_Yhat = wt_a*weights + wt_b

wt_SSres = ((Y - wt_Yhat)**2).sum()
wt_Rsq = 1. - wt_SSres/SStot
print("weights only Rsq: " + str(wt_Rsq))

f, axes = plt.subplots(1,1)
sns.scatterplot(x = weights, y = Y, ax = axes)
sns.lineplot(x = weights, y = wt_Yhat, ax = axes)
axes.set(xlabel='weight (lbs)', ylabel='predicted sys blood pressure')
plt.show()

# years vs weight vs blood pressure (2d)
w = np.linalg.solve(X.T @ X, X.T @ Y)
Yhat = X @ w

SSres = ((Y - Yhat)**2).sum()
Rsq = 1. - SSres/SStot
print("years and weight Rsq: " + str(Rsq))

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(X[:,0], X[:,1], Y)
ax.scatter(X[:,0], X[:,1], Yhat)
plt.show()

# above is the way that I worked ahead and finished the problem, below is
# what the instructor did for reference/options
# using the data frame to our advantage, pull out columns by using the keys
df['ones'] = 1 # create bias term column (python seems to fill in col)
Y = df['BP']
X = df[['Age', 'Weight', 'ones']]
AgeOnly = df[['Age', 'ones']]
WeightOnly = df[['Weight', 'ones']]
# quiz test (adding a noise feature, what happens?)
noise = [np.random.randn() for i in range(len(Y))]
df['noise'] = noise
Xnoise = df[['Age', 'Weight', 'noise', 'ones']]
# function to calculate Rsq, call to find what gives the best fit
def get_Rsq(X, Y):
    w = np.linalg.solve(X.T @ X, X.T @ Y)
    Yhat = X @ w
    SSres = ((Y - Yhat)**2).sum()
    SStot = ((Y - Y.mean())**2).sum()
    Rsq = 1. - SSres/SStot
    return Rsq

print("Age only R^2: " + str(get_Rsq(AgeOnly, Y)))
print("Weight only R^2: " + str(get_Rsq(WeightOnly, Y)))
print("X (age and weight) R^2: " + str(get_Rsq(X, Y)))
print("Xnoise (age, weight, and noise) R^2: " + str(get_Rsq(Xnoise, Y)))
