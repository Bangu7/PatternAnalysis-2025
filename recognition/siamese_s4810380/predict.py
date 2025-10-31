############################
# Example usage with results
############################

import argparse
import torch
import visual
import pandas as pd
from modules import SiameseNetwork, BinaryClassifier
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import MelPredictDataset

def make_predictions(test_loader, siamese, classifier, device):
    """ Make predictions on given dataset.

        @params:
            test_loader: the dataloader for testing data
            siamese: the Siamese Network model for embeddings
            classifier: the Binary Classifier used for predictions
            device: the device being used

        @returns:
            y_pred: a dataframe of img names with their predictions
    """
    siamese.eval()
    classifier.eval()

    # Run Predictions on all data
    y_pred = []
    with torch.no_grad():
        for imgs, img_names in test_loader:
            imgs = imgs.to(device)
            
            embeddings = siamese(imgs)

            # Make predictions on embeddings with 0.6 threshold
            outputs = classifier(embeddings)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs > 0.6).astype(int)
            y_pred.extend(zip(img_names, preds, probs))

    return y_pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        type=str,
        required=True,
        help="Model name of directory to be saved/loaded."
    )

    parser.add_argument(
        "-meta",
        type=str,
        required=True,
        help="Metadata file path. Containing names of images"
    )

    parser.add_argument(
        "-img",
        type=str,
        required=True,
        help="Imgs file path to be predicted."
    )

    parser.add_argument(
        "-out",
        type=str,
        default="pred.csv",
        help="Output file path for predictions."
    )
    
    parser.add_argument(
        "-name",
        type=str,
        default="isic_id",
        help="Column name of image."
    )

    args = parser.parse_args()
    model_name = args.m
    model_fp = f"./model/{model_name}"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    df = pd.read_csv(args.meta, index_col=0)
    img_names = df[args.name].tolist()

    test_data = MelPredictDataset(df, args.img, transform)

    test_loader = DataLoader(test_data, batch_size=16, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    siamese = SiameseNetwork().to(device)
    siamese.load_state_dict(torch.load(f"{model_fp}/siamese.pt", map_location=device))
    classifier = BinaryClassifier().to(device)
    classifier.load_state_dict(torch.load(f"{model_fp}/classifier.pt", map_location=device))

    # Make predictions and save them to given file
    y_pred = make_predictions(test_loader, siamese, classifier, device)
    pred_df = pd.DataFrame(y_pred, columns=['Image_name', 'Prediction', 'Probability'])
    pred_df.to_csv(args.out, index=False)
    print("Predictions saved")

    visual.plot_images(pred_df[pred_df['Prediction']==0], pred_df[pred_df['Prediction']==1], args.img)

