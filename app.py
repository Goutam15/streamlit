import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os

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

# Function to load the trained model
@st.cache_resource
def load_model():
    input_size = 4  # Number of input features
    num_classes = 2  # Binary classification
    model = MentalHealthNet(input_size, num_classes)

    # Check if the model file exists
    model_path = "model/mental_health_model.pth"
    if not os.path.exists(model_path):
        st.error("Model file not found. Please train and upload it.")
        return None

    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

# Load the model
model = load_model()

# Streamlit UI
st.title("🧠 Mental Health Comfort Level Predictor")

st.write("Adjust the sliders below and click **Predict** to see the results.")

# Input sliders
feature1 = st.slider("Feature 1", 0.0, 1.0, 0.5)
feature2 = st.slider("Feature 2", 0.0, 1.0, 0.5)
feature3 = st.slider("Feature 3", 0.0, 1.0, 0.5)
feature4 = st.slider("Feature 4", 0.0, 1.0, 0.5)

if model and st.button("🔮 Predict"):
    # Prepare input data
    features = np.array([[feature1, feature2, feature3, feature4]])
    input_tensor = torch.tensor(features, dtype=torch.float32)

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1).numpy()
        predicted_class = np.argmax(probabilities)

    # Display result
    st.subheader("📊 Prediction Result")
    st.write(f"**Predicted Comfort Level:** {'Comfortable 😊' if predicted_class == 1 else 'Not Comfortable 😞'}")
    st.write(f"**Confidence Scores:** {probabilities.tolist()[0]}")

st.write("Model trained using a simple neural network. Future improvements will include better datasets and optimizations.")
