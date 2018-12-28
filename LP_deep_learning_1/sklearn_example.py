from process import get_data # use the ecommerce data

from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle

# load ecommerce data
X, T = get_data()

# split into train and test sets
X, T = shuffle(X, T)
Ntrain = int(.7*X.shape[0])
Xtrain, Ttrain = X[:Ntrain], T[:Ntrain]
Xtest, Ttest = X[Ntrain:], T[Ntrain:]

# create neural network (two hidden layers of size 20)
model = MLPClassifier(hidden_layer_sizes=(20, 20), max_iter=2000)

# train the network
model.fit(Xtrain, Ttrain)

# print train and test accuracy
train_accuracy = model.score(Xtrain, Ttrain)
test_accuracy = model.score(Xtest, Ttest)
print('train acc:', train_accuracy, 'test acc:', test_accuracy)
