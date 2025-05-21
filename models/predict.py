from bib import *
from data.data_loader import get_data_loaders

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt



def predict(val_loader, true_labels):
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load('./resnet18.pth', map_location=device))
    model = model.to(device)
    model.eval()

    predictions = []
    print("Generating predictions...")

    with torch.no_grad():
        for batch_idx, inputs in enumerate(val_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs).cpu().numpy()
            predictions.extend(preds)

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(val_loader):
                print(f"  Processed batch {batch_idx+1}/{len(val_loader)}")


        # Write binary predictions to a text file
        with open('./label_val.txt', 'w') as f:
          for pred in predictions:
            f.write(f"{int(pred > 0.5)}\n")

        print(" Predictions saved to './label_val.txt'")




def optimize_threshold(predictions, true_labels):
    fpr, tpr, thresholds = roc_curve(true_labels, predictions)
    far = fpr
    frr = 1 - tpr
    hter = (far + frr) / 2

    optimal_idx = hter.argmin()
    optimal_threshold = thresholds[optimal_idx]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, hter, label='HTER')
    plt.axvline(optimal_threshold, color='r', linestyle='--', label=f'Optimal Threshold: {optimal_threshold:.2f}')
    plt.xlabel('Threshold')
    plt.ylabel('HTER')
    plt.title('HTER vs Threshold')
    plt.legend()
    plt.grid(True)
    plt.show()

    return optimal_threshold

