import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import nest_asyncio

# Define input data format
class MentalHealthInput(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float  # Adjust based on dataset features

# Define the neural network model
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

# Function to train the model
def train_model():
    # Generate synthetic dataset (Replace with actual dataset)
    np.random.seed(42)
    num_samples = 500
    num_features = 4
    X_train = np.random.rand(num_samples, num_features)
    y_train = np.random.randint(0, 2, size=(num_samples,))

    # Convert to torch tensors
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

    # Save trained model
    torch.save(model.state_dict(), "mental_health_model.pth")
    print("Model training complete and saved.")

# Train the model
train_model()

# Load the trained model
input_size = 4  # Update based on the actual number of features
num_classes = 2
model = MentalHealthNet(input_size, num_classes)
model.load_state_dict(torch.load("mental_health_model.pth", map_location=torch.device("cpu")))
model.eval()

# Initialize FastAPI app
app = FastAPI()

@app.post("/predict")
def predict(input_data: MentalHealthInput):
    """Predicts mental health discussion comfort level based on user input."""
    
    # Convert input data to tensor
    features = np.array([[input_data.feature1, input_data.feature2, input_data.feature3, input_data.feature4]])
    input_tensor = torch.tensor(features, dtype=torch.float32)

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1).numpy()
        predicted_class = np.argmax(probabilities)

    # Construct response
    response = {
        "prediction": int(predicted_class),
        "confidence_scores": probabilities.tolist()[0],
        "message": f"The model predicts that the user {'would' if predicted_class == 1 else 'would not'} feel comfortable discussing mental health at work."
    }
    return response


@app.get("/")
def home():
    return {"message": "Welcome to the Mental Health Prediction API! Use /predict to get results."}

# Run FastAPI app
nest_asyncio.apply()
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
