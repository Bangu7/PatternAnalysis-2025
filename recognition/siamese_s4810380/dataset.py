######################
# Contains Data Loader
######################
from torch.utils.data import Dataset
import os
import pandas as pd
from torchvision.io import decode_image

class MelSiameseDataset(Dataset):
    """Dataset for pairing melanoma images for Siamese Network."""

    def __init__(self, metadata, img_dir, transform=None):
        self.img_labels = pd.read_csv(metadata, index_col=0)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f"{self.img_labels.iloc[idx, 0]}.jpg")
        img1 = decode_image(img_path)
        label = self.img_labels.iloc[idx, 2]
        if self.transform:
            img1 = self.transform(img1)
        return img1, label


# Example usage, and testing.
if __name__ == "__main__":
    img_dir = "./data/train-image/image"
    metadata = "./data/train-metadata.csv"

    dataset = MelSiameseDataset(metadata, img_dir)
