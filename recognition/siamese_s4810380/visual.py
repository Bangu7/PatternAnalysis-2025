import os
import torch
import random
import numpy as np
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

def loss_plot(train_losses, val_losses, fp, show=False):
    """ Loss plotting of training and validation loss of models.

        @params:
            train_losses: the training losses of model, 1 per epoch
            val_losses: the validation losses of model, 1 per epoch
            n_epochs: number of epochs for the losses
            fp: the file path to save the figure (no file extension)
            show: if to show the plot as well as save
    """
    # Clear plot
    plt.clf()
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Contrastive Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.savefig(f"{fp}.png")
    if show:
        plt.show()

def score_plot(values, title, fp, show=False):
    """ Used to show one line of scores.
        
        @params:
            values: the values to be plotted
            title: title of plot
            fp: the file path to save the figure (no file extension)
            show: if to show the plot as well as save
    """
    plt.clf()
    plt.plot(range(1, len(values)+1), values)
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title(title)
    plt.savefig(f"{fp}.png")
    if show:
        plt.show()

def roc_auc(dataloader, classifier_model, siamese_model, device, fp, show=False):
    """ Generates and saves a ROC curve plot for the Classifier model.

        @params:
            dataloader: the dataloader of images to predict
            classifier_model: the trained classifier to predict embedded images
            siamese_model: the trained network to pass images through
            device: the device being used for processing
            fp: the file path to save the figure (no file extension)
            show: if to show the plot as well as save
    """
    plt.clf()

    classifier_model.eval()
    siamese_model.eval()

    true_labels = []
    probs = []

    # Use num_workers=0 for no memory issues
    dataloader = DataLoader(dataloader.dataset, batch_size=dataloader.batch_size, num_workers=0, shuffle=False)

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.numpy()

            # Produce embeddings to predict
            embeddings = siamese_model(imgs)

            # Predict class probabilities
            outputs = classifier_model(embeddings)
            prob = torch.sigmoid(outputs).cpu().numpy().flatten()
            true_labels.extend(labels)
            probs.extend(prob)

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(true_labels, probs)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(f"{fp}.png")
    if show:
        plt.show()

def tsne_embeddings(dataloader, siamese_model, device, fp, show=False):
    """ The TSNE embeddings plot of the Siamese network.

        @params:
            dataloader: the dataloader to grab embeddings from
            siamese_model: the trained network to pass images through
            device: the device being used for processing
            fp: the file path to save the figure (no file extension)
            show: if to show the plot as well as save
    """
    plt.clf()
    # Don't Train model
    siamese_model.eval()
    
    embeddings_list = []
    labels_list = []

    # Use num_workers=0 due to memory issues
    dataloader = DataLoader(
        dataloader.dataset,
        batch_size=dataloader.batch_size,
        num_workers=0,
        shuffle=False
    )

    # Retrieve embeddings from all images
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)

            # Retrieve embeddings and add to list
            embeddings = siamese_model(imgs)
            embeddings_list.append(embeddings.cpu().numpy())
            labels_list.append(labels.numpy())

    # Concatenate all embeddings and labels
    embeddings = np.concatenate(embeddings_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    # Apply t-SNE to embeddings
    # https://www.datacamp.com/tutorial/introduction-t-sne
    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(embeddings)

    # Create a scatter plot of the 2D t-SNE result
    plt.figure(figsize=(10, 8))
    
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, alpha=0.6)
    handles, _ = scatter.legend_elements()
    labels = ['Benign', 'Malignant']

    plt.legend(handles, labels, title="Class")
    plt.title("t-SNE Visualisation of Siamese Network Embeddings")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.savefig(f"{fp}.png")
    if show:
        plt.show()

def classifier_confusion(dataloader, classifier_model, siamese_model,
                         device, fp, show=False):
    """ Produces a confusion matrix for the Classifier model.

        @params:
            dataloader: the dataloader of images to predict
            classifier_model: the trained classifier to predict embedded images
            siamese_model: the trained network to pass images through
            device: the device being used for processing
            fp: the file path to save the figure (no file extension)
            show: if to show the plot as well as save
    """
    plt.clf()
    # Set both models into evaluation
    classifier_model.eval()
    siamese_model.eval()
    
    true_labels = []
    predictions = []

    # Use num_workers=0 here for visualisation
    dataloader = DataLoader(dataloader.dataset, batch_size=dataloader.batch_size, num_workers=0, shuffle=False)

    with torch.no_grad():
        # Go through all images and predict class
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.numpy()

            # Produce embeddings to predict
            embeddings = siamese_model(imgs)

            # Predict image class based on embeddings
            outputs = classifier_model(embeddings)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            # Predict based on 0.6 threshold
            preds = (probs > 0.6).astype(int)
            true_labels.extend(labels)
            predictions.extend(preds)
    
    # Compute the confusion matrix
    cm = confusion_matrix(true_labels, predictions)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    labels = ['Benign', 'Malignant']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix of Classifier")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.savefig(f"{fp}.png")
    if show:
        plt.show()


def plot_images(images0, images1, img_path):
    """ Plots image predictions of 3 from each class.

        @params:
            images0: the dataframe of images predicted as 0
            images0: the dataframe of images predicted as 1
            img_path: the directory of images
    """
    # Randomly select 3 images
    images0_sample = random.sample(list(images0.iterrows()), 3)
    images1_sample = random.sample(list(images1.iterrows()), 3)

    fig, axes = plt.subplots(2, 3, figsize=(15, 6))

    # Plot 3 images from class 0 (Benign)
    for i, (idx, row) in enumerate(images0_sample):
        img_name = row['Image_name']
        img = Image.open(os.path.join(img_path, img_name+".jpg"))
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"{img_name}\nBenign\nProb: {1-row['Probability']:.2f}")
        axes[0, i].axis('off')

    # Plot 3 images from class 1 (Malignant)
    for i, (idx, row) in enumerate(images1_sample):
        img_name = row['Image_name']
        img = Image.open(os.path.join(img_path, img_name+".jpg"))
        axes[1, i].imshow(img)
        axes[1, i].set_title(f"{img_name}\nMalignant\nProb: {row['Probability']:.2f}")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()
