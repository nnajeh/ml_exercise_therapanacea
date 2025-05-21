from bib import *
from models import train_model
from data.data_loader import get_data_loaders
from models.predict import predict, optimize_threshold
import numpy as np
import torch




if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # Dataset paths
    train_img_dir = './ml_exercise_therapanacea/train_img/'
    val_img_dir = './ml_exercise_therapanacea/val_img/'
    train_labels_file = './ml_exercise_therapanacea/label_train.txt'
    val_labels_file = './ml_exercise_therapanacea/label_val_gt.txt'  # ground truth

    # Dataloaders
    train_loader, val_loader = get_data_loaders(train_img_dir, train_labels_file, val_img_dir)

    # Train model
    train_model(train_loader)

    # Predict probabilities
    predictions = predict(val_loader)

    # Load true labels
    true_labels = np.loadtxt(val_labels_file)

    # Optimize threshold
    optimal_threshold = optimize_threshold(predictions, true_labels)

    # Apply threshold
    binary_predictions = [int(p > optimal_threshold) for p in predictions]

    # Save predictions
    with open('./label_val.txt', 'w') as f:
        for pred in binary_predictions:
            f.write(f"{pred}\n")

    print(f" Predictions saved using optimal threshold: {optimal_threshold:.4f}")
