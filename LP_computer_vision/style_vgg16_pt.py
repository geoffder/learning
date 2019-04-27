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
    X = X.view(X.size()[1], -1)  # flatten spatial dimensions
    return (X @ X.t()) / X.numel()


class StyleTransferLoss(torch.autograd.Function):

    def __init__(self, alpha=.5, beta=.5):
        super(StyleTransferLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, X_content, X_style, content_out, style_out):
        content_loss = torch.mean((X_content - content_out)**2)
        style_loss = torch.sum(torch.stack(
            [torch.sum((Gram(x) - Gram(s))**2)
             for x, s in zip(X_style, style_out)]
        ))
        # print("content loss:", content_loss.data,
        #       "style loss:", style_loss.data)
        output = self.alpha*content_loss + self.beta*style_loss
        return output


class StyleTransferLossMod(nn.Module):

    def __init__(self, alpha=.5, beta=.5):
        super(StyleTransferLossMod, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, X_content, X_style, content_out, style_out):
        content_loss = torch.mean((X_content - content_out)**2)
        style_loss = torch.sum(torch.stack(
            [torch.sum((Gram(x) - Gram(s))**2)
             for x, s in zip(X_style, style_out)]
        ))
        output = self.alpha*content_loss + self.beta*style_loss
        return output


class StyleVGG16(object):
    """
    Trying out tranfer learning with VGG16. This class uses the same structure
    I have used for my own networks, but instead uses a pre-trained build of
    the VGG network with the relevant components switched out to fit the data.
    """
    def __init__(self, content_layer=17, style_layers=[0, 5, 10, 17, 24]):
        self.model = self.build()
        self.model.to(device)
        self.content_layer = content_layer
        self.style_layers = set(style_layers)  # style outputs

    def build(self):
        # load pre-trained vgg (only use convolutional layers)
        vgg = models.vgg16(pretrained=True)
        # vgg = models.vgg16_bn(pretrained=True)
        # take conv layers out of the nn.Sequential that holds them
        vgg_conv = nn.ModuleList(
            list(list(vgg.children())[0].children())[:-1]
        )

        # freeze covolutional layers
        for layer in vgg_conv.children():
            for param in layer.parameters():
                param.requires_grad = False

        # change all max pooling layers to avg pooling
        for i, layer in enumerate(vgg_conv.children()):
            if isinstance(layer, nn.MaxPool2d):
                vgg_conv[i] = nn.AvgPool2d(2, stride=2, padding=0)
            elif isinstance(layer, nn.ReLU):
                vgg_conv[i] = nn.ReLU(inplace=False)

        return vgg_conv

    def forward(self, X, content, style):
        """
        Feed constructed, content, and style images through the network,
        saving the representations at the specified layers
        (self.content_layer, self.style_layers).
        """
        X_style, style_out = [], []
        for i, layer in enumerate(self.model.children()):
            X = layer(X)
            content = layer(content)
            style = layer(style)
            if i in self.style_layers:
                X_style.append(X)
                style_out.append(style)
            if i == self.content_layer:
                content_out = content
                X_content = X  # take final output to compare with content

        return X_content, X_style, content_out, style_out

    def generate(self, content, style, alpha=.5, beta=.5, lr=1e-3, epochs=10,
                 print_every=50, lbfgs=False):
        self.lbfgs = lbfgs  # whether to use L-BFGS optimizer

        # re-shape images for PyTorch
        content = content.reshape(1, *content.shape)
        style = style.reshape(1, *style.shape)
        # send data to GPU
        self.X = torch.randn(content.shape, requires_grad=True, device='cuda')
        self.content = torch.from_numpy(content).float().to(device)
        self.style = torch.from_numpy(style).float().to(device)

        # self.loss = StyleTransferLoss(alpha=alpha, beta=beta)
        self.loss = StyleTransferLossMod(alpha=alpha, beta=beta).to(device)
        if not lbfgs:
            self.optimizer = optim.Adam([self.X], lr=lr)
        else:
            self.optimizer = optim.LBFGS([self.X], lr=1)

        costs = []
        for i in range(epochs):
            self.X.data.clamp_(0, 1)
            cost = self.paint()
            if i % print_every == 0:
                print("epoch: %d, cost: %f" % (i, cost))
            costs.append(cost)
            if lbfgs:
                self.optimizer.step(self.paint)

        painting = self.X.detach().cpu().numpy().reshape(*content.shape[1:])
        print("Painting value range: %.2f -> %.2f"
              % (painting.min(), painting.max()))
        painting = (painting-painting.min()) / (painting.max()-painting.min())

        content = content.reshape(*content.shape[1:])
        style = style.reshape(*style.shape[1:])

        # plot cost and accuracy progression
        fig, axes = plt.subplots(2, 2)
        axes[0, 0].imshow(content.transpose(1, 2, 0), interpolation='bessel')
        axes[0, 1].imshow(style.transpose(1, 2, 0), interpolation='bessel')
        axes[1, 0].plot(costs)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Cost')
        axes[1, 1].imshow(painting.transpose(1, 2, 0), interpolation='bessel')
        # fig.tight_layout()
        plt.show()

        return painting

    def paint(self):
        self.model.train()  # set the model to testing mode
        self.optimizer.zero_grad()  # Reset gradient

        # Forward
        X_content, X_style, content_out, style_out = self.forward(
            self.X, self.content, self.style)
        output = self.loss.forward(X_content, X_style, content_out, style_out)

        # Backward
        output.backward()
        if not self.lbfgs:
            self.optimizer.step()  # Update parameters

        return output.item()


def main():
    # load content image
    # content = Image.open('../large_files/test_images/toby.jpg')
    # content = Image.open('../large_files/test_images/annes_700.jpg')
    content = Image.open('../large_files/test_images/schwabes.jpg')
    content = np.array(content).transpose(2, 0, 1) / 255
    # load style image
    # style = Image.open('../large_files/test_images/starry_night_800.jpg')
    style = Image.open('../large_files/test_images/Claude_Monet_400.jpg')
    style = np.array(style).transpose(2, 0, 1) / 255

    # build network
    vgg = StyleVGG16(content_layer=10)
    # transer style
    painting = vgg.generate(
        content, style, alpha=1, beta=5, lr=1e-1, epochs=5, print_every=2,
        lbfgs=True
    )
    painting = (painting.transpose(1, 2, 0)*255).astype(np.uint8)
    img = Image.fromarray(painting)
    img.save('painting.jpg')


if __name__ == '__main__':
    main()
