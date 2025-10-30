######################
# Contains custom Dataset
######################

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
import random
import numpy as np

class MelSiameseDataset(Dataset):
    """ Dataset for pairing melanoma images for Siamese Network. """

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
                along with a integer representing if they are a match.
        """
        # Randomly pick which type of pair
        if torch.rand(1) < 0.5:
            match = 0
            # Select opposite label
            img_name1 = random.choice(self.labels1.iloc[:, 0].values)
            img_name2 = random.choice(self.labels2.iloc[:, 0].values)
        else:
            match = 1
            # Select same label
            label_choice = random.choice([0, 1])
            class_df = self.labels1 if label_choice == 0 else self.labels2
            img_name1, img_name2 = random.sample(list(class_df.iloc[:, 0].values), 2)
        
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
                metadata: the fp for metadata of each image
                img_dir: the location of all images for the dataset
                transform: the image transform to the dataset
        """
        self.img_labels = pd.read_csv(metadata, index_col=0)
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

def create_dataloaders(metadata, img_dir, train_transform, val_test_transform, 
                       batch_size=16, num_workers=4, train_split=0.7,
                       val_split=0.1, test_split=0.2, seed=7):
    """ Function to automatically retrive dataloaders and seperate
        the given dataset into train, validation, and test sets.

        @params:
            metadata: fp of metadata for images
            img_dir: fp of images
            train_tranform: image transform to training data
            val_test_transform: image transform to validation and test data
            batch_size: batch size for dataloaders
            num_workers: number of workers for each dataloader
            train_split: the amount of data assigned to training data
            val_split: the amount of data assigned to validation data
            test_split: the amount of data assigned to testing data
            seed: random seed to be used for reproducability of data

        @return:
            Returns two tuples of three dataloaders generated using given params.
            Training, validation, testing for siamese and classifier respectively.
    """
    # Create the two Datasets from given data
    full_siamese_data = MelSiameseDataset(metadata, img_dir)
    full_classifier_data = MelClassifierDataset(metadata, img_dir)

    size = len(full_siamese_data)
    torch.manual_seed(seed)

    # Calculate sizes of each dataset and shuffle indices
    indices = torch.randperm(size).tolist()
    train_size = int(train_split * size)
    val_size = int(val_split * size)
    test_size = size - train_size - val_size

    # Split indices for train, val, and test
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Create subsets of data using the pytorch Subset
    # https://stackoverflow.com/questions/47432168/taking-subsets-of-a-pytorch-dataset
    subsets = {}
    for name, idx in zip(['train', 'val', 'test'], [train_indices, val_indices, test_indices]):
        subsets[f"{name}_siamese"] = Subset(full_siamese_data, idx)
        subsets[f"{name}_classifier"] = Subset(full_classifier_data, idx)

    # Assign transforms
    subsets['train_siamese'].dataset.transform = train_transform
    subsets['val_siamese'].dataset.transform = val_test_transform
    subsets['test_siamese'].dataset.transform = val_test_transform

    subsets['train_classifier'].dataset.transform = train_transform
    subsets['val_classifier'].dataset.transform = val_test_transform
    subsets['test_classifier'].dataset.transform = val_test_transform

    # AI Prompt: How can I implement an undersampler for the training dataLoader? (provided partial code)
    def make_sampler(subset):
        labels = subset.dataset.img_labels.iloc[subset.indices]['target'].values
        class_weights = 1.0 / np.bincount(labels)
        sample_weights = np.array([class_weights[t] for t in labels])
        return WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True
        )

    classifier_sampler = make_sampler(subsets['train_classifier'])

    # Create dataloaders for siamese
    train_loader_siamese = DataLoader(subsets['train_siamese'], batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader_siamese = DataLoader(subsets['val_siamese'], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader_siamese = DataLoader(subsets['test_siamese'], batch_size=batch_size, shuffle=False, num_workers=num_workers)

    train_loader_classifier = DataLoader(subsets['train_classifier'], batch_size=batch_size, sampler=classifier_sampler, num_workers=num_workers)
    val_loader_classifier = DataLoader(subsets['val_classifier'], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader_classifier = DataLoader(subsets['test_classifier'], batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return (train_loader_siamese, val_loader_siamese, test_loader_siamese), \
           (train_loader_classifier, val_loader_classifier, test_loader_classifier)
