#########################################
# Training, validating, testing and saving
#########################################

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

def contrastive_loss(y, d, margin=1.0):
    """ The loss function utilised """
    return torch.mean(y * d.pow(2) + (1 - y) * torch.clamp(margin - d, min=0).pow(2))

def train_siamese(train_load, val_load, model, model_name, device, n_epochs=5, show_plot=True):
    """ The main training loop used to obtain a model.

        @params:
            train_load: training set dataloader
            val_load: validation set dataloader
            model: the model to be trained
            model_name: this will be used in the fp for the saved model
            device: what device is being used
            show_plot: boolean value if the plot of loss across epochs
                should be displayed
    """
    optimiser = Adam(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0.0

        for i, (img1, img2, label) in enumerate(train_load):
            if not i % 50 and i != 0:
                print(f"Train batch [{i}/{len(train_load)}], avg loss: {total_train_loss/i:.4f}")
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            # forward pass
            emb1, emb2 = model.forward(img1, img2)
            distance = F.pairwise_distance(emb1, emb2)

            # compute loss
            loss = contrastive_loss(label, distance)

            # backpropagation
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_load)
        print(f"Epoch [{epoch+1}/{n_epochs}] - Training loss: {avg_train_loss:.4f}")

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for img1, img2, label in val_load:
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                emb1, emb2 = model(img1, img2)
                distance = F.pairwise_distance(emb1, emb2)
                loss = contrastive_loss(label, distance)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_load)
        print(f"Epoch [{epoch+1}/{n_epochs}] - Validation loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{model_name}.pt")
            print("Saved new best model")

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

    if show_plot:
        plt.plot(range(1, n_epochs+1), train_losses, label='Train Loss')
        plt.plot(range(1, n_epochs+1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Contrastive Loss')
        plt.title('Training & Validation Loss')
        plt.legend()
        plt.show()

def eval_siamese(test_load, model, model_name, device):
    """ Evaluates a model on given test data and outputs performance. 

        @params:
            test_load: the testing dataloader
            model: model to be evaluated
            model_name: the name of the model to be loaded
            device: device being used
    """
    model.load_state_dict(torch.load(f"{model_name}.pt"))
    model.eval()

    test_loss = 0.0
    with torch.no_grad():
        for img1, img2, label in test_load:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            emb1, emb2 = model(img1, img2)
            distance = F.pairwise_distance(emb1, emb2)
            loss = contrastive_loss(label, distance)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_load)
    print(f"Test loss: {avg_test_loss:.4f}")

def train_classifier(train_loader, val_loader, classifier_name, siamese_model_name, device, n_epochs=5):
    siamese = SiameseNetwork().to(device)
    siamese.load_state_dict(torch.load(f"{siamese_model_name}.pt", map_location=device))

    for param in siamese.base.parameters():
          param.requires_grad = False

    # Create classifier, optimiser and criterion
    classifier = BinaryClassifier().to(device)
    optimiser = Adam(classifier.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    # Run main training loop
    for epoch in range(n_epochs):
        classifier.train()
        total_train_loss = 0

        for imgs, labels in train_loader:
            # Alter label for BCE
            imgs, labels = imgs.to(device), labels.to(device).float().unsqueeze(1)

            with torch.no_grad():
                embeddings = siamese(imgs)
            outputs = classifier(embeddings)
            loss = criterion(outputs, labels)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        classifier.eval()
        total_val_loss = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device).float().unsqueeze(1)
                with torch.no_grad():
                    embeddings = siamese(imgs)
                outputs = classifier(embeddings)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{n_epochs}] - Train loss: {avg_train_loss:.4f} | Val loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(classifier.state_dict(), f"{classifier_name}.pt")
            print("Saved new best classifier")

    if show_plot:
        plt.plot(range(1, n_epochs+1), train_losses, label='Train Loss')
        plt.plot(range(1, n_epochs+1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Contrastive Loss')
        plt.title('Training & Validation Loss')
        plt.legend()
        plt.show()

def eval_classifier(test_loader, siamese, classifier, device):
    siamese.eval()
    classifier.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            
            embeddings = siamese(imgs)

            outputs = classifier(embeddings)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            y_true.extend(labels.numpy())
            y_pred.extend(preds)

    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_true, y_pred):.4f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_dir = "./data/train-image/image"
    metadata = "./data/train-metadata.csv"
    siamese_model_name = "siamese_model"
    classifier_model_name = "classifier_model"

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)), # Make sure images are 224x224
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
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
        num_workers=1,
        train_split=0.7,
        val_split=0.1,
        test_split=0.2,
        seed=7
    )

    n_epochs = 5
    model = SiameseNetwork().to(device)
    train_siamese(siamese_train, siamese_val, model, siamese_model_name, device, n_epochs)
    #eval_siamese(test_load, model, model_name, device)

    n_epochs = 5
    train_classifier(classifier_train, classifier_val, classifier_model_name,
                     siamese_model_name, device, n_epochs)

    siamese = SiameseNetwork().to(device)
    siamese.load_state_dict(torch.load(f"{siamese_model_name}.pt", map_location=device))
    classifier = BinaryClassifier().to(device)
    classifier.load_state_dict(torch.load(f"{classifier_model_name}.pt", map_location=device))

    # Freeze Parameters of base
    # https://discuss.pytorch.org/t/how-the-pytorch-freeze-network-in-some-layers-only-the-rest-of-the-training/7088
    for param in siamese.base.parameters():
        param.requires_grad = False

    eval_classifier(classifier_test, siamese, classifier, device)
