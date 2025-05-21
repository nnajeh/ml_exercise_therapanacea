
from data.data_loader import get_data_loaders

def train_model(train_loader, num_epochs=10, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), 'models/resnet18.pth')

def predict(val_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load('models/resnet18.pth'))
    model = model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad():
        for inputs in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs).cpu().numpy()
            predictions.extend(preds)

    with open('output/label_val.txt', 'w') as f:
        for pred in predictions:
            f.write(f"{int(pred > 0.5)}\n")

if __name__ == '__main__':
    train_loader, val_loader = get_data_loaders('train_img', 'label_train.txt', 'val_img')
    train_model(train_loader)
    predict(val_loader)
