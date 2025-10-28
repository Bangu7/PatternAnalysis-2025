######################
# Contains custom Dataset
######################

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import random

class MelSiameseDataset(Dataset):
    """Dataset for pairing melanoma images for Siamese Network."""

    def __init__(self, metadata, img_dir, transform=None):
        """ Initialises Dataset.
            
            @params:
                metadata: the fp for metadata of each image
                img_dir: the location of all images for the dataset
                transform: the image transform to the dataset
        """
        self.img_labels = pd.read_csv(metadata, index_col=0)

        # Split into positive and negative classes
        self.labels1 = self.img_labels[self.img_labels['target'] == 0]
        self.labels2 = self.img_labels[self.img_labels['target'] == 1]

        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        """ Returns length/size of dataset """
        return len(self.img_labels)

    def __getitem__(self, idx):
        """ Gives the image pair for given index.

            @params:
                idx: index of image to be paired

            @return:
                Tuple of decided image with its randomly decided pair
                alon with a integer representing if they are a match.
        """
        # Retrieve image data of given index
        label = self.img_labels.iloc[idx, 2]
        img_name1 = self.img_labels.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, f"{img_name1}.jpg")
        img1 = Image.open(img_path).convert("RGB")

        # Randomly pick which pair
        if torch.rand(1) < 0.5:
            match = 0
            # Select opposite label
            diff_class = self.labels2 if label == 0 else self.labels1
            img_name2 = random.choice(diff_class.iloc[:, 0].values)
        else:
            match = 1
            # Select same label
            same_class = self.labels1 if label == 0 else self.labels2
            img_name2 = random.choice(same_class.iloc[:, 0].values)
        
        # Retrieve the second images based on randomly generated index
        img_path2 = os.path.join(self.img_dir, f"{img_name2}.jpg")
        img2 = Image.open(img_path2).convert("RGB")

        # Transform images if given a transform
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(match, dtype=torch.float32)

def create_dataloaders(metadata, img_dir, train_transform, val_test_trans, 
                       batch_size=16, num_workers=4, train_split=0.7,
                       val_split=0.1, test_split=0.2, seed=7):
    """ Function to automatically retrive dataloaders and seperate
        the given dataset into train, validation, and test sets.

        @params:
            metadata: fp of metadata for images
            img_dir: fp of images
            train_tranform: image transform to training data
            val_test_trans: image transform to validation and test data
            batch_size: batch size for dataloaders
            num_workers: number of workers for each dataloader
            train_split: the amount of data assigned to training data
            val_split: the amount of data assigned to validation data
            test_split: the amount of data assigned to testing data
            seed: random seed to be used for reproducability of data

        @return:
            Returns tuple of three dataloaders generated using given params.
            Training, validation, testing.
    """
    full_data = MelSiameseDataset(metadata, img_dir)
    size = len(full_data)

    # Calculate sizes of each dataset
    train_size = int(train_split * size)
    val_size = int(val_split * size)
    test_size = size - train_size - val_size

    torch.manual_seed(seed)

    # https://discuss.pytorch.org/t/torch-utils-data-dataset-random-split/32209/4
    train_data, val_data, test_data = random_split(
        full_data,
        [train_size, val_size, test_size]
    )

    # Applies transforms to subsets of data.
    train_data.dataset.transform = train_transform
    val_data.dataset.transform = val_test_trans
    test_data.dataset.transform = val_test_trans

    # Uses datasets to make dataloaders.
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_loader, val_loader, test_loader
