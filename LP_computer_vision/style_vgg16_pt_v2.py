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


class StyleTransferLoss(nn.Module):
    """
    Custom function that combines Content Loss (MSE between source content
    and contstruct convolutional representation from a particular layer) and
    Style Loss (MSE between Gram matrices [Features x Features] of source style
    and construct convolutional representations from a set of layers).

    alpha and beta are used to weight the importance of content and style loss
    respectively.
    """
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
        output = self.alpha*content_loss + self.beta*style_loss
        return output


class StyleVGG16(object):
    """
    Style-transfer using pre-trained VGG16 network.
    """
    def __init__(self, content_layer=17, style_layers=[0, 5, 10, 17, 24],
                 batchnorm=False):
        self.model = self.build(batchnorm)
        self.model.to(device).eval()  # send to GPU and set to evaluation mode
        self.content_layer = content_layer
        self.style_layers = set(style_layers)  # style outputs

    def build(self, batchnorm):
        # load pre-trained vgg (only use convolutional layers)
        if not batchnorm:
            vgg = models.vgg16(pretrained=True)  # with batchnorm
        else:
            vgg = models.vgg16_bn(pretrained=True)  # with batchnorm

        # take conv layers out of the nn.Sequential that holds them
        vgg_conv = nn.ModuleList(
            list(list(vgg.children())[0].children())[:-1]
        )

        # freeze covolutional and batchnorm layers
        for param in vgg_conv.parameters():
            param.requires_grad = False

        # change all maxpool layers to avgpool and make ReLU not inplace.
        for i, layer in enumerate(vgg_conv.children()):
            if isinstance(layer, nn.MaxPool2d):
                vgg_conv[i] = nn.AvgPool2d(2, stride=2, padding=0)
            elif isinstance(layer, nn.ReLU):
                # not using sequential, so inplace would not work as intended
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
            # transform X, content, and style with forward() of current layer
            X = layer(X)
            content = layer(content)
            style = layer(style)
            # store convolutional representations of target layers
            if i in self.style_layers:
                X_style.append(X)
                style_out.append(style.detach())  # not needed in graph
            if i == self.content_layer:
                content_out = content.detach()    # not needed in graph
                X_content = X

        return X_content, X_style, content_out, style_out

    def generate(self, content, style, alpha=.5, beta=.5, lr=1, epochs=10,
                 print_every=50):

        # re-shape images for PyTorch
        content = content.reshape(1, *content.shape)
        style = style.reshape(1, *style.shape)
        # send data to GPU
        self.X = torch.randn(content.shape, requires_grad=True, device='cuda')
        self.content = torch.from_numpy(content).float().to(device)
        self.style = torch.from_numpy(style).float().to(device)

        self.loss = StyleTransferLoss(alpha=alpha, beta=beta).to(device)
        self.optimizer = optim.LBFGS([self.X], lr=lr)

        costs = []
        for i in range(epochs):
            # clip to float image range 0 -> 1
            self.X.data.clamp_(0, 1)
            # execute round of L-BFGS optimization with self.paint as closure()
            cost = self.optimizer.step(self.paint)
            # print progress
            if i % print_every == 0:
                print("epoch: %d, cost: %f" % (i, cost))
            costs.append(cost)
        print('')  # line break

        painting = self.X.detach().cpu().numpy().reshape(*content.shape[1:])
        print("Painting value range: %.2f -> %.2f"
              % (painting.min(), painting.max()))
        # painting = (
        #   painting-painting.min()) / (painting.max()-painting.min())
        painting = painting.clip(0, 1)

        content = content.reshape(*content.shape[1:])
        style = style.reshape(*style.shape[1:])

        # plot original content, style, cost progression, and created image
        fig, axes = plt.subplots(2, 2)
        axes[0, 0].imshow(content.transpose(1, 2, 0), interpolation='bessel')
        axes[0, 1].imshow(style.transpose(1, 2, 0), interpolation='bessel')
        axes[1, 0].plot(costs)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Cost')
        axes[1, 1].imshow(painting.transpose(1, 2, 0), interpolation='bessel')
        plt.show()

        return painting

    def paint(self):
        """
        Closure function for L-BFGS optimizer. Note the lack of
        optimizer.step() here as there would be with SGD optimizers. Instead,
        this function is to optimizer.step() in the training/optimization loop.
        """
        self.optimizer.zero_grad()  # Reset gradient

        # Forward
        X_content, X_style, content_out, style_out = self.forward(
            self.X, self.content, self.style)
        output = self.loss.forward(X_content, X_style, content_out, style_out)

        # Backward
        output.backward()

        return output.item()  # cost


def main():
    # load content image
    content = Image.open('../large_files/test_images/toby_500.jpg')
    # content = Image.open('../large_files/test_images/annes_700.jpg')
    # content = Image.open('../large_files/test_images/schwabes.jpg')
    content = np.array(content).transpose(2, 0, 1) / 255
    # load style image
    style = Image.open('../large_files/test_images/starry_night_600.jpg')
    # style = Image.open('../large_files/test_images/Claude_Monet_400.jpg')
    style = np.array(style).transpose(2, 0, 1) / 255

    # build network
    vgg = StyleVGG16(content_layer=17, batchnorm=True)
    # transer style
    painting = vgg.generate(
        content, style, alpha=1, beta=400, lr=1, epochs=5, print_every=2
    )
    painting = (painting.transpose(1, 2, 0)*255).astype(np.uint8)
    img = Image.fromarray(painting)
    img.save('painting.jpg')


if __name__ == '__main__':
    main()
