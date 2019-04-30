import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import resize

import torch
# from torch import nn
from torchvision import models

# use GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


class ClassMapResNet50(object):
    """
    Make predictions and produce class activation maps using pre-trained
    and frozen ResNet50.
    """
    def __init__(self):
        self.model = self.build()
        self.fc_weights = self.model._modules['fc'].weight.cpu().numpy()
        self.model.eval().to(device)  # only using evaluation mode

    def build(self):
        resnet = models.resnet50(pretrained=True)

        # freeze all layers
        for param in resnet.parameters():
            param.requires_grad = False

        # change final pooling to global (allow all image sizes)
        # this was giving the wrong sized output (dimensions flipped)
        # resnet._modules['avgpool'] = nn.AdaptiveAvgPool2d(1)

        return resnet

    def forward(self, X):
        for i, child in enumerate(self.model.children()):
            X = child(X)
            if i == 7:
                conv_out = X
            elif i == 8:
                # instead of adaptive pooling layer, which was giving me a
                # 2048x1 tensor rather than 1x2048 (mul to 2048x1000 weights)
                X = torch.mean(X, dim=(2, 3))
        return conv_out, X

    def map(self, images):

        classmaps, predictions = [], []
        for img in images:
            # convert to tensor and include batch dimension
            imtensor = torch.from_numpy(
                img.reshape(1, *img.shape)).float().to(device)

            # forward pass, returning
            fmaps, logits = self.features_and_logits(imtensor)

            # predict class label and apply softmax to get % confidence score
            p = np.argmax(logits)  # predicted class index
            confs = np.exp(logits) / np.exp(logits).sum(keepdims=True)

            # get feature weights corresponding to the predicted class
            w = self.fc_weights[p, :].reshape(-1, 1, 1)
            heat = np.sum(np.squeeze(fmaps)*w, axis=0)  # weighted sum of maps
            heat = (heat-heat.min()) / (heat.max()-heat.min())  # normalize

            # store upscaled heat-maps and and predicted labels with confidence
            classmaps.append(
                resize(heat, img.shape[1:], mode='constant',
                       anti_aliasing=True)
            )
            predictions.append([lookup[p], confs[p]])  # string label and conf

        # display
        for img, heat, label in zip(images, classmaps, predictions):
            plt.imshow(img.transpose(1, 2, 0))  # re-shape to RGB last
            plt.imshow(heat, alpha=.6, cmap='jet')
            plt.title("%s: %.3f" % (label[0], label[1]))
            plt.show()

    def features_and_logits(self, X):
        with torch.no_grad():
            features, logits = self.forward(X)
        # return convolutional output and logits as numpy arrays
        return features.data.cpu().numpy(), logits.data.cpu().numpy()[0]


def main():
    images = [
        Image.open('../large_files/test_images/toby_crop.jpg'),
        Image.open('../large_files/test_images/hotdogs.jpg'),
        Image.open('../large_files/test_images/annes_700.jpg'),
    ]
    images = [np.array(img).transpose(2, 0, 1) / 255
              for img in images]
    resn = ClassMapResNet50()
    resn.map(images)


if __name__ == '__main__':
    # mapping of label indices (1000 output classes) with their english names
    lookup = pd.read_json('ImageNet1000_labels.json', typ='series')
    main()
