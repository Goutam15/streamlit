import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# Load dataset
file_path = "Mental Health Data.csv"  # Update path if needed
df = pd.read_csv(file_path)

# Map categorical values to binary
binary_mapping = {"Yes": 1, "No": 0, "Maybe": 0}  
df["Target"] = df["Would you feel comfortable discussing a mental health disorder with your coworkers?"].map(binary_mapping)
df["Feature1"] = df["Does your employer provide mental health benefits as part of healthcare coverage?"].map(binary_mapping)
df["Feature2"] = df["Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?"].map(binary_mapping)
df["Feature3"] = df["Do you have a family history of mental illness?"].map(binary_mapping)
df["Feature4"] = df["Have you ever sought treatment for a mental health issue from a mental health professional?"].map(binary_mapping)

# Drop missing values
df = df.dropna(subset=["Target", "Feature1", "Feature2", "Feature3", "Feature4"])

# Split data
X = df[["Feature1", "Feature2", "Feature3", "Feature4"]].values
y = df["Target"].values

# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Define model
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

# Initialize and train model
input_size = 4
num_classes = 2
model = MentalHealthNet(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), "mental_health_model.pth")
print("Model training complete. Saved as 'mental_health_model.pth'.")
