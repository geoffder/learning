import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch import nn
from torch import optim
from torchvision import models

# use GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


def Gram(X):
    "Return (C x C) Gram Matrix"
    X = X.view(X.size()[0], -1)  # flatten spatial dimensions
    return (X @ X.t()) / X.numel()


class StyleTransferLoss(torch.autograd.Function):

    def __init__(self, alpha=.5, beta=.5):
        super(StyleTransferLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, X_out, content_out, style_out):
        content_loss = torch.mean((X_out[-1] - content_out)**2)
        style_loss = torch.sum(torch.stack(
            [torch.mean((Gram(x) - Gram(s))**2)
             for x, s in zip(X_out[:-1], style_out)]
        ))
        output = self.alpha*content_loss + self.beta*style_loss
        return output


class StyleTransferLossMod(nn.Module):

    def __init__(self, alpha=.5, beta=.5):
        super(StyleTransferLossMod, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, X_out, content_out, style_out):
        content_loss = torch.mean((X_out[-1] - content_out)**2)
        style_loss = torch.sum(torch.stack(
            [torch.mean((Gram(x) - Gram(s))**2)
             for x, s in zip(X_out[:-1], style_out)]
        ))
        output = self.alpha*content_loss + self.beta*style_loss
        return output


class StyleVGG16(object):
    """
    Trying out tranfer learning with VGG16. This class uses the same structure
    I have used for my own networks, but instead uses a pre-trained build of
    the VGG network with the relevant components switched out to fit the data.
    """
    def __init__(self):
        self.model = self.build()
        self.model.to(device)
        self.out_layers = set([0, 5, 10, 17, 24])  # style outputs

    def build(self):
        # load pre-trained vgg (only use convolutional layers)
        vgg = models.vgg16(pretrained=True)
        vgg_conv = nn.Sequential(*list(vgg.children())[:-1])  # skip last pool

        # freeze covolutional layers
        for param in vgg_conv.parameters():
            param.requires_grad = False

        # change all max pooling layers to avg pooling
        for i, layer in enumerate(vgg_conv.children()):
            if isinstance(layer, nn.MaxPool2d):
                vgg_conv[i] = nn.AvgPool2d(2, stride=2, padding=0)

        return vgg_conv

    def forward(self, X, content, style):
        X_out, style_out = [], []
        for i, layer in enumerate(self.model.children()):
            X = layer(X)
            content = layer(content)
            style = layer(style)
            if i in self.out_layers:
                X_out.append(X)
                style_out.append(style)
        X_out.append(X)  # take final output to compare with content

        # for debuging
        # painting = content.detach().cpu().numpy()
        # painting = np.mean(painting, axis=1)
        # painting = painting.reshape(*painting.shape[1:])
        # if painting.min() < 0:
        #     painting += -1*painting.min()
        # painting = painting / painting.max()
        # plt.imshow(painting)
        # plt.show()

        return X_out, content, style_out

    def generate(self, content, style, alpha=.5, beta=.5, lr=1e-3, epochs=10):

        # re-shape images for PyTorch
        content = content.reshape(1, *content.shape)
        style = style.reshape(1, *style.shape)
        # send data to GPU
        self.X = torch.randn(content.shape, requires_grad=True, device='cuda')
        self.content = torch.from_numpy(content).float().to(device)
        self.style = torch.from_numpy(style).float().to(device)

        # self.loss = StyleTransferLoss(alpha=alpha, beta=beta)
        self.loss = StyleTransferLossMod(alpha=alpha, beta=beta).to(device)
        self.optimizer = optim.Adam([self.X], lr=lr)
        # self.optimizer = optim.LBFGS([self.X], lr=1)

        costs = []
        for i in range(epochs):
            cost = self.paint()
            print("epoch: %d, cost: %f" % (i, cost))
            costs.append(cost)
            # self.optimizer.step(self.paint)

        painting = self.X.detach().cpu().numpy().reshape(*content.shape[1:])
        if painting.min() < 0:
            painting += -1*painting.min()
        painting = painting / painting.max()
        # plot cost and accuracy progression
        fig, axes = plt.subplots(1, 2)
        axes[0].plot(costs)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Cost')
        axes[1].imshow(painting.transpose(1, 2, 0))
        fig.tight_layout()
        plt.show()

        return painting

    def paint(self):
        self.model.train()  # set the model to testing mode
        self.optimizer.zero_grad()  # Reset gradient

        # Forward
        X_out, content_out, style_out = self.forward(
            self.X, self.content, self.style)
        output = self.loss.forward(X_out, content_out, style_out)

        # Backward
        output.backward()
        self.optimizer.step()  # Update parameters

        return output.item()


def main():
    # load content image
    content = Image.open('../large_files/test_images/toby.jpg')
    content = np.array(content).transpose(2, 0, 1) / 255
    # load style image
    style = Image.open('../large_files/test_images/starry_night.jpg')
    style = np.array(style).transpose(2, 0, 1) / 255

    # build network
    vgg = StyleVGG16()
    # transer style
    vgg.generate(content, style, alpha=1, beta=0, lr=1e-1, epochs=100)


if __name__ == '__main__':
    main()
