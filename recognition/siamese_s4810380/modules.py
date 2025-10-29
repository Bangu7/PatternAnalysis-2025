#####################
# Components of model
#####################

import torch
import torch.nn as nn
from torchvision.models import resnet18

class SiameseNetwork(nn.Module):
    """ Siamese Netowrk using ResNet as a base model """

    def __init__(self):
        """ Initialise model """
        super().__init__()
        base = resnet18(weights=None, progress=False)
        self.base = nn.Sequential(*list(base.children()))[:-1]
        self.embedding_dim = 512

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
        emb1 = self.base(emb1)
        emb1 = emb1.view(emb1.size(0), -1) # Flatten

        if img2 is None:
            return emb1

        # Forward pass for img2
        emb2 = self.base(img2)
        emb2 = emb2.view(emb2.size(0), -1) # Flatten
        return emb1, emb2

class BinaryClassifier(nn.Module):
    """ Binary classifer based on Siamese embeddings """

    def __init__(self, emb_dim=512):
        """ Initialise Classifier model """
        super().__init__()
        # Simple classification network
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, 256),
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
