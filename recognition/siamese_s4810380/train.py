#########################################
# Training, validating, testing and saving
#########################################

import os
import visual
import argparse
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.nn import TripletMarginLoss
from modules import SiameseNetwork, BinaryClassifier
from dataset import create_dataloaders
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def contrastive_loss(y, d, margin=1.3):
    """ Gives the contrastive loss of a list of distances given a margin
        and the ground truths.

        @params:
            y: Ground truth labels.
            d: A tensor of the pairswise distances between the vectors
                being compared.
            margin: Threshold for how far dissimilar pairs should be.

        @return:
            The mean contrastive loss across the given pairs.
    """
    # https://www.geeksforgeeks.org/deep-learning/loss-functions-in-deep-learning/
    # AI aided conversion to pytorch over numpy.
    return 0.5 * torch.mean(y * d.pow(2) + (1 - y) * torch.clamp(margin - d, min=0).pow(2))

def train_siamese(train_loader, val_loader, model_fp, device, n_epochs, verbose=False):
    """ The main training loop used to obtain a model.

        @params:
            train_loader: training set dataloader
            val_loader: validation set dataloader
            model_fp: file path to save model
            device: what device is being used
            n_epochs: number of epochs to be trained for
            verbose: boolean value if the train batch loss sohuld be
                outputted to terminal
    """
    model = SiameseNetwork().to(device)
    optimiser = Adam(model.parameters(), lr=1e-4)

    # Set initial values
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    no_improve = 0

    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0.0

        # Training loop
        for i, (img1, img2, label) in enumerate(train_loader):
            if not i % 10 and i != 0 and verbose:
                print(f"Train batch [{i}/{len(train_loader)}], avg loss: {total_train_loss/i:.4f}")
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            # forward pass
            emb1, emb2 = model(img1, img2)

            # compute loss
            distance = F.pairwise_distance(emb1, emb2)
            loss = contrastive_loss(label, distance)

            # backpropagation
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Set to evaluation for validation data
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for img1, img2, label in val_loader:
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                emb1, emb2 = model(img1, img2)
                distance = F.pairwise_distance(emb1, emb2)
                loss = contrastive_loss(label, distance)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{n_epochs}] - Train loss: {avg_train_loss:.4f} | Val loss: {avg_val_loss:.4f}")

        # Check if model improved
        if avg_val_loss < best_val_loss:
            no_improve = 0
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{model_fp}/siamese.pt")
            print("Saved new best model")
        else:
            no_improve += 1

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if no_improve > 4:
            print("No improvents in past 5 epochs")
            break

    visual.loss_plot(train_losses, val_losses, f"{model_fp}/siamese_loss")

def train_classifier(train_loader, val_loader, model_fp, device, n_epochs, verbose=False):
    """ Train the binary classifier on the siamese network.
        
        @params:
            train_loader: the training dataloader
            val_loader: the validation dataloader
            model_fp: the location of the models to be loaded and trained
            device: the device being used
            n_epochs: the number of epochs to train classifier
            verbose: if set will print average loss during epochs
    """
    siamese = SiameseNetwork().to(device)
    siamese.load_state_dict(torch.load(f"{model_fp}/siamese.pt", map_location=device))

    # Freeze Siamese network layers
    for param in siamese.base.parameters():
          param.requires_grad = False

    # Create classifier, optimiser and criterion
    classifier = BinaryClassifier().to(device)
    optimiser = Adam(classifier.parameters(), lr=5e-5)
    criterion = nn.BCEWithLogitsLoss()

    # Initialise trackers
    best_val_loss = float('inf')
    best_val_auc = 0.0
    train_losses = []
    val_losses = []
    val_aucs = []
    val_f1s = []
    no_improve = 0

    # Run main training loop
    for epoch in range(n_epochs):
        classifier.train()
        total_train_loss = 0

        all_labels = []
        all_preds = []
        all_probs = []

        for i, (imgs, labels) in enumerate(train_loader):
            if not i % 10 and i != 0 and verbose:
                print(f"Train batch [{i}/{len(train_loader)}], avg loss: {total_train_loss/i:.4f}")

            # Alter label for BCE
            imgs, labels = imgs.to(device), labels.to(device).float().unsqueeze(1)

            with torch.no_grad():
                embeddings = siamese(imgs)
            outputs = classifier(embeddings)
            loss = criterion(outputs, labels)

            # Backpropagation
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Run validation loop
        classifier.eval()
        total_val_loss = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                # Alter label for BCE
                imgs, labels = imgs.to(device), labels.to(device).float().unsqueeze(1)

                # Compute loss from emebeddings
                with torch.no_grad():
                    embeddings = siamese(imgs)
                outputs = classifier(embeddings)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                # Make prediction with 0.6 threshold
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.6).float()

                all_labels.append(labels.cpu())
                all_preds.append(preds.cpu())
                all_probs.append(probs.cpu())

        # Add new performance to arrays
        labels_np = torch.cat(all_labels).numpy()
        preds_np = torch.cat(all_preds).numpy()
        probs_np = torch.cat(all_probs).numpy()

        val_f1 = f1_score(labels_np, preds_np)
        val_auc = roc_auc_score(labels_np, probs_np)

        val_f1s.append(val_f1)
        val_aucs.append(val_auc)

        # Calculate average loss
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{n_epochs}] - Train loss: {avg_train_loss:.4f} | Val loss: {avg_val_loss:.4f}")
        print(f"Epoch [{epoch+1}/{n_epochs}] - F1: {val_f1:.4f} | AUC: {val_auc:.4f}")

        # Check if model is improving
        if val_auc > best_val_auc:  
            no_improve = 0
            best_val_auc = val_auc
            torch.save(classifier.state_dict(), f"{model_fp}/classifier.pt")
            print("Saved new best model")
        else:
            no_improve += 1

        if no_improve > 4:
            print("No improvents in past 5 epochs")
            break

    # Output plots of training performance
    visual.loss_plot(train_losses, val_losses, f"{model_fp}/classifier_loss")
    visual.score_plot(val_aucs, "Validation AUC", f"{model_fp}/val_auc")
    visual.score_plot(val_f1s, "Validation F1", f"{model_fp}/val_f1")

def eval_siamese(test_loader, model, device):
    """ Evaluates a model on given test data and outputs performance. 

        @params:
            test_loader: the testing dataloader
            model: model to be evaluated
            device: device being used
    """
    model.eval()

    test_loss = 0.0
    with torch.no_grad():
        for img1, img2, label in test_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            emb1, emb2 = model(img1, img2)
            distance = F.pairwise_distance(emb1, emb2)
            loss = contrastive_loss(label, distance)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print("\nSiamese")
    print(f"Test loss: {avg_test_loss:.4f}")

def eval_classifier(test_loader, siamese, classifier, device):
    """ Evaluates classifier model on given data.

        @params:
            test_loader: dataloader to be tested
            siamese: the Siamese network model the classifier was
                trained on
            classifier: the classifier model
            device: device being used
    """
    siamese.eval()
    classifier.eval()

    y_true = []
    y_pred = []
    y_prob = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            
            embeddings = siamese(imgs)

            outputs = classifier(embeddings)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs > 0.6).astype(int)
            y_true.extend(labels.numpy())
            y_pred.extend(preds)
            y_prob.extend(probs)

    print("\nClassifier")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_true, y_prob):.4f}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train and evaluate Siamese network and BinaryClassifer"
    )

    # Model name arguments
    parser.add_argument(
        "-m",
        type=str,
        default="0",
        help="Model name of directory to be saved/loaded."
    )

    # Options
    parser.add_argument(
        "-e",
        type=int,
        default=5,
        help="Number of epochs to train each model."
    )
    parser.add_argument(
        "-v",
        action="store_true",
        help="If set, enables verbose output during training."
    )
    parser.add_argument(
        "-train",
        action="store_true",
        help="If set, trains models."
    )
    parser.add_argument(
        "-eval",
        action="store_true",
        help="If set, performs test evaluation on models."
    )   
    parser.add_argument(
        "-p",
        action="store_true",
        help="If set, plots performance of models and saves."
    )   

    args = parser.parse_args()

    # Assign args
    model_name = args.m
    n_epochs = args.e
    verbose = args.v
    training = args.train
    evaluation = args.eval
    plots = args.p

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_dir = "./data/train-image/image"
    metadata = "./data/train-metadata.csv"

    # Make model directory to store models if not already there
    model_fp = f"./model/{model_name}"
    os.makedirs(model_fp, exist_ok=True)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)), # Make sure images are 224x224
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ), # ImageNet normalisation (Used for ResNet)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    (siamese_train, siamese_val, siamese_test), \
    (classifier_train, classifier_val, classifier_test)= create_dataloaders(
        metadata,
        img_dir,
        train_transform,
        val_transform,
        batch_size=16,
        num_workers=4,
        seed=7
    )

    if training:
        train_siamese(siamese_train, siamese_val, model_fp, device, n_epochs, verbose)
        train_classifier(classifier_train, classifier_val, model_fp, device, n_epochs, verbose)

    siamese = SiameseNetwork().to(device)
    siamese.load_state_dict(torch.load(f"{model_fp}/siamese.pt", map_location=device))
    classifier = BinaryClassifier().to(device)
    classifier.load_state_dict(torch.load(f"{model_fp}/classifier.pt", map_location=device))

    # Freeze Parameters of base
    # https://discuss.pytorch.org/t/how-the-pytorch-freeze-network-in-some-layers-only-the-rest-of-the-training/7088
    for param in siamese.base.parameters():
        param.requires_grad = False

    if evaluation: 
        eval_siamese(siamese_test, siamese, device)
        eval_classifier(classifier_test, siamese, classifier, device)

    if plots:
        visual.tsne_embeddings(classifier_test, siamese, device, f"{model_fp}/tsne")
        visual.classifier_confusion(classifier_test, classifier, siamese, device, f"{model_fp}/cm")
        visual.roc_auc(classifier_test, classifier, siamese, device, f"{model_fp}/roc")
