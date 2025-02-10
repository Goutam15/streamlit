import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

# Define neural network
class MentalHealthNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MentalHealthNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# Train model
def train_model():
    # Generate synthetic dataset (replace with real data)
    np.random.seed(42)
    num_samples = 500
    num_features = 4
    X_train = np.random.rand(num_samples, num_features)
    y_train = np.random.randint(0, 2, size=(num_samples,))

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    # Initialize model
    input_size = num_features
    num_classes = 2
    model = MentalHealthNet(input_size, num_classes)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train model
    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    # Create model directory if not exists
    os.makedirs("model", exist_ok=True)

    # Save trained model
    torch.save(model.state_dict(), "model/mental_health_model.pth")
    print("Model training complete and saved.")

# Run training
if __name__ == "__main__":
    train_model()
