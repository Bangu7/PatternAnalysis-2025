import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

def loss_plot(train_losses, val_losses, n_epochs, fp):
    """ Loss plotting of training and validation loss of models.

        @params:
            train_losses: the training losses of model, 1 per epoch
            val_losses: the validation losses of model, 1 per epoch
            n_epochs: number of epochs for the losses
            fp: the file path to save the figure (no file extension)
    """
    plt.plot(range(1, n_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, n_epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Contrastive Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.savefig(f"{fp}.png")

def tsne_embeddings(dataloader, siamese_model, device, fp):
    """ The TSNE embeddings plot of the Siamese network.

        @params:
            dataloader: The dataloader to grab embeddings from
            siamese_model: the trained network to pass images through
            device: the device being used for processing
            fp: the file path to save the figure (no file extension)
    """
    # Don't Train model
    siamese_model.eval()
    
    embeddings_list = []
    labels_list = []

    # Use num_workers=0 due to memory issues
    dataloader = DataLoader(dataloader.dataset, batch_size=dataloader.batch_size, num_workers=0, shuffle=False)

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
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)

    # Create a scatter plot of the 2D t-SNE result
    plt.figure(figsize=(10, 8))
    
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, alpha=0.6)
    handles, _ = scatter.legend_elements()
    legend_labels = ['Benign (0)', 'Melanoma (1)']

    plt.legend(handles, legend_labels, title="Class")
    plt.title("t-SNE Visualisation of Siamese Network Embeddings")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.savefig(f"{fp}.png")

def classifier_confusion(dataloader, classifier_model, siamese_model, device, fp):
    """ Produces a confusion matrix for the Classifier model.

        @params:
            dataloader: The dataloader of images to predict
            classifier_model: the trained classifier to predict embedded images
            siamese_model: the trained network to pass images through
            device: the device being used for processing
            fp: the file path to save the figure (no file extension)
    """
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
            # Predict based on 0.5 threshold
            preds = (probs > 0.5).astype(int)
            true_labels.extend(labels)
            predictions.extend(preds)
    
    # Compute the confusion matrix
    cm = confusion_matrix(true_labels, predictions)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    labels = ['Benign', 'Melanoma']
    sns.heatmap(cm, annot=True, fmt='d', cmap='blue', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix of Classifier")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.savefig(f"{fp}.png")
