#####################
# Components of model
#####################

import torch.nn as nn
from torchvision.models import resnet18

class SiameseNetwork(nn.Module):
    """ Siamese Netowrk using ResNet as a base model """

    def __init__(self):
        """ Initialise model """
        super().__init__()
        self.base = resnet18(weights=None)

    def forward(self, img1, img2):
        """ Forward processing of image pair.

            @params:
                img1: A tensor of the first image,
                img2: A tensor of the second image.

            @return:
                Pair of image tensors after forward pass.
        """
        out1 = self.base(img1)
        out2 = self.base(img2)
        return out1, out2
