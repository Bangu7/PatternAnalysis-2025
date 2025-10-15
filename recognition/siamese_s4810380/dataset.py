######################
# Contains custom Dataset
######################

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class MelSiameseDataset(Dataset):
    """Dataset for pairing melanoma images for Siamese Network."""

    def __init__(self, metadata, img_dir, transform=None):
        self.img_labels = pd.read_csv(metadata, index_col=0)

        # Split into positive and negative classes
        self.labels1 = self.img_labels[self.img_labels['target'] == 0]
        self.labels2 = self.img_labels[self.img_labels['target'] == 1]

        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f"{self.img_labels.iloc[idx, 0]}.jpg")
        img1 = Image.open(img_path)
        label = self.img_labels.iloc[idx, 2]

        # Randomly pick which pair
        if torch.rand(1) < 0.5:
            match = 0
            # Select opposite label
            if label:
                img_path2 = os.path.join(self.img_dir, f"{self.labels1.iloc[idx, 0]}.jpg")
            else:
                img_path2 = os.path.join(self.img_dir, f"{self.labels2.iloc[idx, 0]}.jpg")
        else:
            match = 1
            # Select same label
            if not label:
                img_path2 = os.path.join(self.img_dir, f"{self.labels1.iloc[idx, 0]}.jpg")
            else:
                img_path2 = os.path.join(self.img_dir, f"{self.labels2.iloc[idx, 0]}.jpg")
        
        img2 = Image.open(img_path2)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(match)

# Example usage, and testing.
if __name__ == "__main__":
    img_dir = "./data/train-image/image"
    metadata = "./data/train-metadata.csv"

    dataset = MelSiameseDataset(metadata, img_dir)
    print(dataset[10])
