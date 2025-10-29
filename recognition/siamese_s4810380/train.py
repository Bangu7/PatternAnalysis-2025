#########################################
# Training, validating, testing and saving
#########################################

import torch
from torch.optim.adam import Adam
from torch.nn import TripletMarginLoss
from modules import SiameseNetwork
from dataset import create_dataloaders
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt

def contrastive_loss(y, d, margin=1.0):
    """ The loss function utilised """
    return torch.mean(y * d.pow(2) + (1 - y) * torch.clamp(margin - d, min=0).pow(2))

def training(train_load, val_load, model, model_name, device, show_plot=True):
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
        total_loss = 0.0

        for i, (img1, img2, label) in enumerate(train_load):
            if not i % 50 and i != 0:
                print(f"Train batch [{i}/{len(train_load)}], avg loss: {total_loss/i:.4f}")
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

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_load)
        print(f"Epoch [{epoch+1}/{n_epochs}] - Training loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img1, img2, label in val_load:
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                emb1, emb2 = model(img1, img2)
                distance = F.pairwise_distance(emb1, emb2)
                loss = contrastive_loss(label, distance)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_load)
        print(f"Epoch [{epoch+1}/{n_epochs}] - Validation loss: {avg_val_loss:.4f}")

        # ---- Save best model ----
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
        plt.ylabel('Loss')
        plt.title('Training & Validation Loss')
        plt.legend()
        plt.show()

def eval_test(test_load, model, model_name, device):
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

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_dir = "./data/train-image/image"
    metadata = "./data/train-metadata.csv"
    n_epochs = 2
    model_name = "best_siamese_model"

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

    train_load, val_load, test_load = create_dataloaders(
        metadata,
        img_dir,
        train_transform,
        val_transform,
        batch_size=16,
        num_workers=8,
        train_split=0.7,
        val_split=0.1,
        test_split=0.2,
        seed=7
    )

    model = SiameseNetwork().to(device)
    training(train_load, val_load, model, model_name, device)
    eval_test(test_load, model, model_name, device)
