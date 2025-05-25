# ✏️ Digit Classifier

A real-time digit recognition web application built with Streamlit and PyTorch. Draw digits (0-9) on an interactive canvas and get instant predictions from a trained Convolutional Neural Network (CNN).


## 🚀 Features

- **Interactive Drawing Canvas**: Draw digits directly in your browser
- **Real-time Prediction**: Get instant classification results
- **CNN Architecture**: Uses a custom 2-layer convolutional neural network
- **Image Preprocessing**: Automatic image processing to match MNIST format
- **Responsive UI**: Clean, user-friendly interface with real-time feedback

## 🏗️ Model Architecture

The CNN model consists of:
- **Conv Layer 1**: 1→32 channels, 3×3 kernel, ReLU activation, MaxPool
- **Conv Layer 2**: 32→64 channels, 3×3 kernel, ReLU activation, MaxPool
- **FC Layer 1**: 3136→128 neurons, ReLU activation, 50% dropout
- **Output Layer**: 128→10 neurons (one for each digit class)

## 📋 Requirements

```
streamlit
matplotlib
opencv-python
streamlit-drawable-canvas
torch
torchvision
numpy
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd mnist-digit-classifier
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the model file**
   - Ensure you have a trained model saved as `mnist_cnn.pth` in the project root
   - The model should be a PyTorch state dict or complete model file

## 🎯 Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser**
   - Navigate to `http://localhost:8501`

3. **Draw and predict**
   - Draw a digit (0-9) on the canvas using your mouse
   - Click the "Predict" button to get the classification result
   - View the processed 28×28 grayscale image used by the model

## 🔧 How It Works

### Image Preprocessing Pipeline
1. **Color Conversion**: RGBA/RGB → Grayscale
2. **Color Inversion**: Black background, white digit (MNIST format)
3. **Resizing**: Canvas image → 28×28 pixels
4. **Normalization**: Pixel values scaled to [0, 1]
5. **Tensor Conversion**: NumPy array → PyTorch tensor with batch dimension

### Model Inference
- Input shape: `(1, 1, 28, 28)` - batch_size, channels, height, width
- Forward pass through CNN layers
- Softmax classification to predict digit class (0-9)

## 📁 Project Structure

```
mnist-digit-classifier/
├── app.py                 # Main Streamlit application
├── mnist_cnn.pth         # Trained PyTorch model (required)
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```

## 🎨 User Interface

- **Left Panel**: Interactive drawing canvas (300×300px)
- **Right Panel**: Prediction results and processed image visualization
- **Sidebar**: Model status and error messages

## ⚠️ Troubleshooting

### Common Issues

1. **"Model file not found" error**
   - Ensure `mnist_cnn.pth` exists in the project root directory
   - Check that the model file is not corrupted

2. **"Please draw a digit first" message**
   - Make sure to draw something on the canvas before clicking "Predict"

3. **Poor prediction accuracy**
   - Try drawing digits more clearly with thicker strokes
   - Ensure digits fill most of the canvas area
   - Draw digits similar to handwritten style (not printed fonts)

### Model Requirements
- The model file must be compatible with the defined `Model` class architecture
- Model should be trained on MNIST or similar 28×28 grayscale digit data
- Expected input format: `torch.FloatTensor` with shape `(batch_size, 1, 28, 28)`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

