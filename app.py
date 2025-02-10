import streamlit as st
import torch
import torch.nn as nn
import numpy as np

# Load trained model
@st.cache_resource
def load_model():
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

    model = MentalHealthNet(4, 2)
    model.load_state_dict(torch.load("mental_health_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# Streamlit UI
st.title("🧠 Mental Health Comfort Level Predictor")

st.markdown(
    "This app predicts whether an individual feels comfortable discussing mental health at work "
    "based on employer policies and personal mental health history."
)

st.subheader("📌 Features Used for Prediction")
st.write("1️⃣ **Does your employer provide mental health benefits?** (Yes/No)")
st.write("2️⃣ **Has your employer discussed mental health policies?** (Yes/No)")
st.write("3️⃣ **Do you have a family history of mental illness?** (Yes/No)")
st.write("4️⃣ **Have you ever sought mental health treatment?** (Yes/No)")

# Input sliders for features
feature1 = st.radio("Does your employer provide mental health benefits?", ["Yes", "No"])
feature2 = st.radio("Has your employer discussed mental health policies?", ["Yes", "No"])
feature3 = st.radio("Do you have a family history of mental illness?", ["Yes", "No"])
feature4 = st.radio("Have you ever sought mental health treatment?", ["Yes", "No"])

# Convert categorical inputs to numerical
feature_mapping = {"Yes": 1, "No": 0}
features = np.array([[feature_mapping[feature1], feature_mapping[feature2], feature_mapping[feature3], feature_mapping[feature4]]])
input_tensor = torch.tensor(features, dtype=torch.float32)

if st.button("🔮 Predict Comfort Level"):
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1).numpy()
        predicted_class = np.argmax(probabilities)

    # Display result
    st.subheader("📊 Prediction Result")
    st.write(f"**Predicted Comfort Level:** {'Comfortable 😊' if predicted_class == 1 else 'Not Comfortable 😞'}")
    st.write(f"**Confidence Scores:** {probabilities.tolist()[0]}")
