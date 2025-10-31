#####################
# Components of model
#####################

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    """ Siamese Netowrk using ResNet as a base model """

    def __init__(self):
        """ Initialise model """
        super().__init__()
        # https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html#torchvision.models.ResNet18_Weights
        base = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Remove final layer that does classification
        self.base = nn.Sequential(*list(base.children()))[:-1]
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

    def forward_once(self, img):
        """ On move forward (one image).

            @params:
                img: image to be embedded

            @returns:
                Embedded image, i.e. after layers and normalisation
        """
        emb = self.base(img)
        emb = self.fc(emb)
        return F.normalize(emb, p=2, dim=1)

    def forward(self, img1, img2=None):
        """ Forward processing of image pair.

            @params:
                img1: A tensor of the first image,
                img2: A tensor of the second image can be null for Classifier.

            @return:
                If img2: Tuple pair of embeddings.
                If no img2: Return embedding of img1. 
        """
        # Forward pass img1
        emb1 = self.forward_once(img1)

        if img2 is None:
            return emb1

        # Forward pass for img2
        emb2 = self.forward_once(img2)

        return emb1, emb2

class BinaryClassifier(nn.Module):
    """ Binary classifer based on Siamese embeddings """

    def __init__(self):
        """ Initialise Classifier model """
        super().__init__()
        # Simple classification network
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, emb):
        """ Forward pass through classifier 
            
            @params:
                emb: the embedding from a Siamese network

            @return:
                Logit from classifier.
        """
        return self.classifier(emb)
