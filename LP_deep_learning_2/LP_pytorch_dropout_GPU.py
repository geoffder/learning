import numpy as np
import matplotlib.pyplot as plt
from LP_util import get_normalized_data

import torch
from torch.autograd import Variable
from torch import optim

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# get the data, same as Theano + Tensorflow examples
# no need to split now, the fit() function will do it
Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()

# get shapes
_, D = Xtrain.shape
K = len(set(Ytrain))

# Note: no need to convert Y to indicator matrix


# the model will be a sequence of layers
# torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True
model = torch.nn.Sequential()
# ANN with layers [784] -> [500] -> [300] -> [10]
# NOTE: the "p" is p_drop, not p_keep
model.add_module("dropout1", torch.nn.Dropout(p=0.2))
model.add_module("dense1", torch.nn.Linear(D, 500))
model.add_module("relu1", torch.nn.ReLU())
model.add_module("dropout2", torch.nn.Dropout(p=0.5))
model.add_module("dense2", torch.nn.Linear(500, 300))
model.add_module("relu2", torch.nn.ReLU())
model.add_module("dropout3", torch.nn.Dropout(p=0.5))
model.add_module("dense3", torch.nn.Linear(300, K))
# Note: no final softmax!
# just like Tensorflow, it's included in cross-entropy function
model.to(device)

# define a loss function
# other loss functions can be found here:
# http://pytorch.org/docs/master/nn.html#loss-functions
loss = torch.nn.CrossEntropyLoss(size_average=True)
# Note: this returns a function!
# e.g. use it like: loss(logits, labels)


# define an optimizer
# other optimizers can be found here:
# http://pytorch.org/docs/master/optim.html
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# define the training procedure
# i.e. one step of gradient descent
# there are lots of steps
# so we encapsulate it in a function
# Note: inputs and labels are torch tensors
def train(model, loss, optimizer, inputs, labels):
    # set the model to training mode
    # because dropout has 2 different modes!
    model.train()

    inputs = Variable(inputs, requires_grad=False)
    labels = Variable(labels, requires_grad=False)

    # Reset gradient
    optimizer.zero_grad()

    # Forward
    logits = model.forward(inputs)
    output = loss.forward(logits, labels)

    # Backward
    output.backward()

    # Update parameters
    optimizer.step()

    # what's the difference between backward() and step()?

    return output.item()


# similar to train() but not doing the backprop step
def get_cost(model, loss, inputs, labels):
    # set the model to testing mode
    # because dropout has 2 different modes!
    model.eval()

    inputs = Variable(inputs, requires_grad=False)
    labels = Variable(labels, requires_grad=False)

    # Forward
    logits = model.forward(inputs)
    output = loss.forward(logits, labels)

    return output.item()


# define the prediction procedure
# also encapsulate these steps
# Note: inputs is a torch tensor
def predict(model, inputs):
    # set the model to testing mode
    # because dropout has 2 different modes!
    model.eval()

    inputs = Variable(inputs, requires_grad=False)
    logits = model.forward(inputs)
    return logits.data.cpu().numpy().argmax(axis=1)


# return the accuracy
# labels is a torch tensor
# to get back the internal numpy data
# use the instance method .numpy()
def score(model, inputs, labels):
    predictions = predict(model, inputs)
    return np.mean(labels.cpu().numpy() == predictions)


# prepare for training loop ###

# convert the data arrays into torch tensors
Xtrain = torch.from_numpy(Xtrain).float().to(device)
Ytrain = torch.from_numpy(Ytrain).long().to(device)
Xtest = torch.from_numpy(Xtest).float().to(device)
Ytest = torch.from_numpy(Ytest).long().to(device)

# training parameters
epochs = 15
batch_size = 200
n_batches = Xtrain.size()[0] // batch_size

# things to keep track of
train_costs = []
test_costs = []
train_accuracies = []
test_accuracies = []

# main training loop
for i in range(epochs):
    cost = 0
    test_cost = 0
    for j in range(n_batches):
        Xbatch = Xtrain[j*batch_size:(j+1)*batch_size]
        Ybatch = Ytrain[j*batch_size:(j+1)*batch_size]
        cost += train(model, loss, optimizer, Xbatch, Ybatch)

        # we could have also calculated the train cost here
        # but I wanted to show you that we could also return it
        # from the train function itself
        train_acc = score(model, Xtrain, Ytrain)
        test_acc = score(model, Xtest, Ytest)
        test_cost = get_cost(model, loss, Xtest, Ytest)

        print("Epoch: %d, cost: %f, acc: %.2f" % (i, test_cost, test_acc))

    # for plotting
    train_costs.append(cost / n_batches)
    train_accuracies.append(train_acc)
    test_costs.append(test_cost)
    test_accuracies.append(test_acc)


# plot the results
plt.plot(train_costs, label='Train cost')
plt.plot(test_costs, label='Test cost')
plt.title('Cost')
plt.legend()
plt.show()

plt.plot(train_accuracies, label='Train accuracy')
plt.plot(test_accuracies, label='Test accuracy')
plt.title('Accuracy')
plt.legend()
plt.show()
