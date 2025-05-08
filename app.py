import streamlit as st
import matplotlib.pyplot as plt
import os
import cv2
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        # Layer 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Layer 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
# Set page config
st.set_page_config(page_title="MNIST Digit Classifier", layout="wide")

# App title and description
st.title("✏️ Real-time Digit Classifier")
st.markdown("""
Draw a digit (0-9) on the canvas below and see the real-time prediction!
""")

@st.cache_resource
def load_model():
    model_path = "mnist_cnn.pth"
    if not os.path.exists(model_path):
        st.sidebar.error(f"❌ Model file '{model_path}' not found!")
        st.sidebar.info("Please upload the model file using the uploader below.")
        return None
    try:
        model = torch.load(model_path, weights_only=False)
        model.eval()  # Set model to evaluation mode
        print("✅ CNN model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Error loading CNN model: {e}")
        return None

model = load_model()

# Function to preprocess drawn image
def preprocess_image(image_data):
    # Convert to grayscale
    if len(image_data.shape) == 3 and image_data.shape[2] == 4:  # RGBA
        image_data = cv2.cvtColor(image_data, cv2.COLOR_RGBA2GRAY)
    elif len(image_data.shape) == 3 and image_data.shape[2] == 3:  # RGB
        image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)
        
    # Invert colors to make digit white on black background
    image_data = 255 - image_data
    
    # Invert colors so digit is white on black background (like MNIST)
    image_data = 255 - image_data
    
    # Resize to 28x28
    image_data = cv2.resize(image_data, (28, 28))
        
    # Normalize pixel values
    processed_img = image_data / 255.0
    
    # Convert to PyTorch tensor with shape (1, 1, 28, 28) for batch, channel, height, width
    model_input = torch.tensor(processed_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    return processed_img, model_input


# Function to make prediction
def predict_digit(image, model):
    processed_img, model_input = preprocess_image(image)
    
    # Make predictions
    model.eval()  # Ensure model is in evaluation mode
    with torch.no_grad():
        outputs = model(model_input)
        _, predicted = torch.max(outputs, 1)

    return predicted, processed_img

# Main content
col1, col2 = st.columns([2, 1])

# Canvas for drawing
with col1:
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=15,
        stroke_color="white",
        background_color="transparent",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    # Add predict button
    predict_button = st.button("Predict")
    # Display prediction
with col2:
    if predict_button:
        if canvas_result.image_data is None:
            st.error("Please draw a digit on the canvas first!")
        elif model is None:
            st.error("Model not loaded. Please check if the model file exists.")
        else:
            # Make prediction
            prediction, processed_img = predict_digit(
                canvas_result.image_data, 
                model, 
            )
            # Display prediction
            st.success(f"Predicted Digit: {prediction}")            
            # Display processed image
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(processed_img, cmap='gray')
            ax.axis('off')
            st.pyplot(fig)
