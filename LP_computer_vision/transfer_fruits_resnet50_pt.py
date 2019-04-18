import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import os
from PIL import Image

import torch
from torch import nn
from torch import optim
from torchvision import models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# use GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


class TransferResNet50(object):
    """
    Trying out tranfer learning with VGG16. This class uses the same structure
    I have used for my own networks, but instead uses a pre-trained build of
    the VGG network with the relevant components switched out to fit the data.

    This version makes use of the pytorch Dataset and Dataloader classes to
    load images from class organized folders (fruits-360 image dataset).
    """
    def __init__(self, dims=None, classes=1000, pretrained=True,
                 freeze_network=False):
        # only specify input dimensions if not [3, 244, 244] (ImageNet dims)
        self.dims = dims  # input dimensions
        self.K = classes  # output classes
        self.pretrained = pretrained
        self.freeze_network = freeze_network
        self.model = self.build()
        self.model.to(device)

    def build(self):
        resnet = models.resnet50(pretrained=self.pretrained)

        # freeze covolutional layers
        if self.freeze_network:
            for param in resnet.parameters():
                param.requires_grad = False

        if self.dims is not None:
            # change to appropriate number of input feature-maps
            krnl = [7, 7]  # default (7x7) with stride=2
            resnet.conv1 = nn.Conv2d(
                self.dims[0], 64, kernel_size=(krnl[0], krnl[1]),
                stride=(2, 2), padding=(krnl[0]//2, krnl[1]//2), bias=False
            )
            resnet.maxpool = nn.MaxPool2d(kernel_size=3)  # default(3x3)

        # change to appropriate number of output classes
        resnet.avgpool = nn.AvgPool2d(kernel_size=7)  # default(7x7)
        resnet.fc = nn.Linear(
            in_features=2048, out_features=self.K, bias=True)

        return resnet

    def fit(self, train_set, test_set, lr=1e-4, epochs=40,
            batch_sz=200, print_every=50):

            train_loader = DataLoader(train_set, batch_size=batch_sz,
                                      shuffle=True, num_workers=2)
            test_loader = DataLoader(test_set, batch_size=batch_sz,
                                     num_workers=2)
            N = train_set.__len__()  # number of samples

            self.loss = nn.CrossEntropyLoss(reduction='mean').to(device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

            n_batches = N // batch_sz
            train_costs, train_accs, test_costs, test_accs = [], [], [], []
            for i in range(epochs):
                cost = 0
                print("epoch:", i, "n_batches:", n_batches)
                for j, batch in enumerate(train_loader):

                    cost += self.train_step(
                        batch['image'].to(device), batch['label'].to(device))

                    if j % print_every == 0:
                        # costs and accuracies for test set
                        test_acc, test_cost = 0, 0
                        for t, testB in enumerate(test_loader, 1):
                            testB_cost, testB_acc = self.cost_and_score(
                                testB['image'].to(device),
                                testB['label'].to(device)
                            )
                            test_cost += testB_cost
                            test_acc += testB_acc
                        test_cost /= t+1
                        test_acc /= t+1
                        # accuracies for train set
                        train_acc = 0
                        for t, trainB in enumerate(train_loader, 1):
                            train_acc += self.score(
                                trainB['image'].to(device),
                                trainB['label'].to(device)
                            )
                        train_acc /= t+1
                        print("cost: %f, acc: %.2f" % (test_cost, test_acc))

                # for plotting
                train_costs.append(cost / n_batches)
                train_accs.append(train_acc)
                test_costs.append(test_cost)
                test_accs.append(test_acc)

            # plot cost and accuracy progression
            fig, axes = plt.subplots(1, 2)
            axes[0].plot(train_costs, label='training')
            axes[0].plot(test_costs, label='validation')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Cost')
            axes[1].plot(train_accs, label='training')
            axes[1].plot(test_accs, label='validation')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            plt.legend()
            fig.tight_layout()
            plt.show()

    def train_step(self, inputs, labels):
        self.model.train()  # set the model to training mode
        self.optimizer.zero_grad()  # Reset gradient

        # Forward
        logits = self.model.forward(inputs)
        output = self.loss.forward(logits, labels)

        # Backward
        output.backward()
        self.optimizer.step()  # Update parameters

        return output.item()

    def get_cost(self, inputs, labels):
        self.model.eval()  # set the model to testing mode
        self.optimizer.zero_grad()  # Reset gradient
        with torch.no_grad():
            # Forward
            logits = self.model.forward(inputs)
            output = self.loss.forward(logits, labels)
        return output.item()

    def predict(self, inputs):
        self.model.eval()
        self.optimizer.zero_grad()  # Reset gradient
        with torch.no_grad():
            logits = self.model.forward(inputs)
        return logits.data.cpu().numpy().argmax(axis=1)

    def score(self, inputs, labels):
        predictions = self.predict(inputs)
        return np.mean(labels.cpu().numpy() == predictions)

    def cost_and_score(self, inputs, labels):
        self.model.eval()  # set the model to testing mode
        self.optimizer.zero_grad()  # Reset gradient
        with torch.no_grad():
            # Forward
            logits = self.model.forward(inputs)
            output = self.loss.forward(logits, labels)
        predictions = logits.data.cpu().numpy().argmax(axis=1)
        acc = np.mean(labels.cpu().numpy() == predictions)
        return output.item(), acc

    def get_confusion_matrix(self, dataset, batch_sz=100):
        """
        Return a confusion matrix of predictions vs targets for a given
        dataset (usually test set). Useful for gaining insight in to which
        classes are giving the model trouble (misclassifying).
        """
        loader = DataLoader(dataset, batch_size=batch_sz, num_workers=2)
        predictions, targets = [], []

        for batch in loader:
            p = self.predict(batch['image'].to(device))
            predictions.append(p)
            targets.append(batch['label'].cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)

        return confusion_matrix(targets, predictions)


class Fruits(Dataset):
    """Fruit-360 Kaggle Dataset"""

    def __init__(self, root_dir, classes, transform=None):
        self.root_dir = root_dir
        self.classes = classes
        self.transform = transform
        self.K = len(classes)
        self.fruit_frame = self.build_lookup(root_dir, classes)

    @staticmethod
    def build_lookup(root_dir, classes):
        "Make a lookup table to class name, label, and filename."
        table = [
            [folder, k, file]
            for k, folder in enumerate(classes)
            for file in [name for name in os.listdir(root_dir+folder+'/')
                         if os.path.isfile(root_dir+folder+'/'+name)]
        ]
        return pd.DataFrame(table)

    def __len__(self):
        "Number of samples in dataset."
        return len(self.fruit_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(
            self.root_dir,
            self.fruit_frame.iloc[idx, 0],  # folder (name of fruit)
            self.fruit_frame.iloc[idx, 2]  # file name
        )

        image = Image.open(img_name)
        label = torch.from_numpy(
            np.array(self.fruit_frame.iloc[idx, 1]))

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.Compose([transforms.ToTensor()])(image)

        sample = {'image': image, 'label': label}
        return sample


def main():
    train_path = '../large_files/fruits-360/Training/'
    test_path = '../large_files/fruits-360/Test/'
    # sub-set of fruits
    # classes = ['Apple Golden 1', 'Avocado', 'Lemon', 'Mango', 'Kiwi',
    #            'Banana', 'Strawberry', 'Raspberry']
    # all of the fruits
    classes = [name for name in os.listdir(train_path)
               if os.path.isdir(train_path+name)]
    
    train_set = Fruits(
        train_path, classes,
        transforms.Compose([
            # transforms.RandomHorizontalFlip(p=.5),
            # transforms.RandomVerticalFlip(p=.5),
            # transforms.RandomRotation(20),
            # transforms.RandomResizedCrop(100),
            transforms.ToTensor()
        ])
    )
    test_set = Fruits(test_path, classes)  # ToTensor() applied by default

    resn = TransferResNet50(
        dims=[3, 100, 100], classes=len(classes),
        pretrained=True, freeze_network=True
    )
    resn.fit(train_set, test_set, lr=1e-3, epochs=10, batch_sz=50)
    # print(resn.get_confusion_matrix(test_set))


if __name__ == '__main__':
    main()
