from bib import *
from data.data_loader import get_data_loaders


def train_model(train_loader, num_epochs=10, learning_rate=0.001):
    
    # Load the pretrained ResNet18 model
    model = models.resnet18(pretrained=True)

    # Replace the final classification layer with a binary output layer
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)

    # Binary classification loss function 
    criterion = nn.BCEWithLogitsLoss()

    # Adam optimizer for weight updates
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # The training mode
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        total_batches = len(train_loader)
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Iterate over mini-batches
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # Move data to the selected device
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item() * inputs.size(0)

            # Display progress every 10 batches
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                print(f"  Batch {batch_idx+1}/{total_batches} - Loss: {loss.item():.4f}")

        # Print average loss for the epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1} completed. Average Loss: {epoch_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), './mymodel.pth')
    print(" Model saved to './mymodel.pth'")

