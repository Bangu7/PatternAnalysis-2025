######################
# Contains custom Dataset
######################

import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

class MelSiameseDataset(Dataset):
    """ Dataset for pairing melanoma images for Siamese Network. """

    def __init__(self, metadata, img_dir, transform=None):
        """ Initialises Dataset.
            
            @params:
                metadata: the label data
                img_dir: the location of all images for the dataset
                transform: the image transform to the dataset
        """
        self.img_labels = metadata.copy()

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
                along with a integer representing if they are a match.
        """
        # Randomly pick which type of pair
        if torch.rand(1) < 0.5:
            # Select opposite labels
            match = 0
            img_name1 = random.choice(self.labels1.iloc[:, 0].values)
            img_name2 = random.choice(self.labels2.iloc[:, 0].values)
        else:
            # Select same label
            match = 1
            if random.random() < 0.3:
                class_df = self.labels1
            else:
                class_df = self.labels2

            img_name1, img_name2 = random.sample(list(class_df.iloc[:, 0].values), 2)
        
        # Open images
        img_path1 = os.path.join(self.img_dir, f"{img_name1}.jpg")
        img_path2 = os.path.join(self.img_dir, f"{img_name2}.jpg")
        img1 = Image.open(img_path1).convert("RGB")
        img2 = Image.open(img_path2).convert("RGB")

        # Transform images if given a transform
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(match, dtype=torch.float32)

class MelClassifierDataset(Dataset):
    """ Dataset for pairing melanoma images for Classifier. """

    def __init__(self, metadata, img_dir, transform=None):
        """ Initialises Dataset.
            
            @params:
                metadata: the metadata of each image
                img_dir: the location of all images for the dataset
                transform: the image transform to the dataset
        """
        self.img_labels = metadata.copy()
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        """ Returns length/size of dataset """
        return len(self.img_labels)

    def __getitem__(self, idx):
        """ Gives the image and label associated with idx

            @params:
                idx: index of image to retrieve

            @return:
                Tuple of image and label.
        """
        # Find image name and convert it to pillow
        img_name = self.img_labels.iloc[idx, 0]
        label = torch.tensor(self.img_labels.iloc[idx, 2], dtype=torch.float32)
        img_path = os.path.join(self.img_dir, f"{img_name}.jpg")
        img = Image.open(img_path).convert("RGB")

        # Transform if given
        if self.transform:
            img = self.transform(img)

        return img, label

class MelPredictDataset(Dataset):
    """ Dataset for predicting images without label. """

    def __init__(self, metadata, img_dir, transform=None):
        """ Initialises Dataset.
            
            @params:
                metadata: the metadata of each image
                img_dir: the location of all images for the dataset
                transform: the image transform to the dataset
        """
        self.img_labels = metadata.copy()
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        """ Returns length/size of dataset """
        return len(self.img_labels)

    def __getitem__(self, idx):
        """ Gives the image associated with idx

            @params:
                idx: index of image to retrieve

            @return:
                img loaded into python with its corresponding name 
        """
        # Find image name and convert it to pillow
        img_name = self.img_labels.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, f"{img_name}.jpg")
        img = Image.open(img_path).convert("RGB")

        # Transform if given
        if self.transform:
            img = self.transform(img)

        return img, img_name

def create_dataloaders(metadata, img_dir, train_transform, val_test_transform, 
                       batch_size=16, num_workers=4, seed=7):
    """ Function to automatically retrive dataloaders and seperate
        the given dataset into train, validation, and test sets.

        @params:
            metadata: fp of metadata for images
            img_dir: fp of images
            train_tranform: image transform to training data
            val_test_transform: image transform to validation and test data
            batch_size: batch size for dataloaders
            num_workers: number of workers for each dataloader
            seed: random seed to be used for reproducability of data

        @return:
            Returns two tuples of three dataloaders generated using given params.
            Training, validation, testing for siamese and classifier respectively.
    """
    df = pd.read_csv(metadata, index_col=0)

    # Seperate classes
    ben_df = df[df['target'] == 0]
    mel_df = df[df['target'] == 1]
    print(f"Total benign: {len(ben_df)}, melanoma: {len(mel_df)}")

    # Subsample benign based on ratio relative to melanoma count
    num_benign = int(len(mel_df)*2.5)
    ben_sampled = ben_df.sample(n=num_benign, random_state=seed, replace=False)
    balanced_df = pd.concat([mel_df, ben_sampled]).sample(frac=1, random_state=seed)

    # Output dataset properties
    print(f"Using benign: {len(ben_sampled)}, melanoma: {len(mel_df)}")

    # Stratified split
    train_df, temp_df = train_test_split(
        balanced_df, test_size=0.3, random_state=seed, stratify=balanced_df['target']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.66, random_state=seed, stratify=temp_df['target']
    )

    # Datasets
    train_siamese = MelSiameseDataset(train_df, img_dir, transform=train_transform)
    val_siamese   = MelSiameseDataset(val_df, img_dir, transform=val_test_transform)
    test_siamese  = MelSiameseDataset(test_df, img_dir, transform=val_test_transform)

    train_classifier = MelClassifierDataset(train_df, img_dir, transform=train_transform)
    val_classifier   = MelClassifierDataset(val_df, img_dir, transform=val_test_transform)
    test_classifier  = MelClassifierDataset(test_df, img_dir, transform=val_test_transform)

    # Dataloaders
    train_loader_siamese = DataLoader(train_siamese, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader_siamese   = DataLoader(val_siamese, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader_siamese  = DataLoader(test_siamese, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    train_loader_classifier = DataLoader(train_classifier, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader_classifier   = DataLoader(val_classifier, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader_classifier  = DataLoader(test_classifier, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return (train_loader_siamese, val_loader_siamese, test_loader_siamese), \
           (train_loader_classifier, val_loader_classifier, test_loader_classifier)
