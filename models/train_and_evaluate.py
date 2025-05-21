from bib import *
from data.data_loader import get_data_loaders



def train_model(train_loader, num_epochs=10, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        total_batches = len(train_loader)
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                print(f"  Batch {batch_idx+1}/{total_batches} - Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1} completed. Average Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), './resnet18.pth')
    print(" Model saved to './resnet18.pth'")

def predict(val_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Predicting on device: {device}")

    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load('./resnet18.pth'))
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

    with open('./label_val.txt', 'w') as f:
        for pred in predictions:
            f.write(f"{int(pred > 0.5)}\n")

    print(" Predictions saved to './label_val.txt'")
