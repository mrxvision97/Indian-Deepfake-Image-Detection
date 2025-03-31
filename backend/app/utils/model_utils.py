# app/utils/model_utils.py

import torch
import timm
import os

# Suppress albumentations update warning (optional; consider updating with: pip install -U albumentations)
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Xception71 Model
def load_xception71_model():
    print("🌟 Initializing Xception71...")
    model = timm.create_model("xception71", pretrained=False, num_classes=1)
    model_path = "Xception71_best.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Please ensure it exists.")
    
    # Load the model weights with CPU mapping
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# Load CustomCNN Model
def load_custom_cnn_model():
    print("🌟 Initializing CustomCNN...")
    class CustomCNN(torch.nn.Module):
        def __init__(self):
            super(CustomCNN, self).__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, kernel_size=3, padding=1), torch.nn.ReLU(inplace=True), torch.nn.BatchNorm2d(64),
                torch.nn.Conv2d(64, 64, kernel_size=3, padding=1), torch.nn.ReLU(inplace=True), torch.nn.BatchNorm2d(64),
                torch.nn.MaxPool2d(kernel_size=2, stride=2), torch.nn.Dropout(0.3),
                torch.nn.Conv2d(64, 128, kernel_size=3, padding=1), torch.nn.ReLU(inplace=True), torch.nn.BatchNorm2d(128),
                torch.nn.Conv2d(128, 128, kernel_size=3, padding=1), torch.nn.ReLU(inplace=True), torch.nn.BatchNorm2d(128),
                torch.nn.MaxPool2d(kernel_size=2, stride=2), torch.nn.Dropout(0.3),
                torch.nn.Conv2d(128, 256, kernel_size=3, padding=1), torch.nn.ReLU(inplace=True), torch.nn.BatchNorm2d(256),
                torch.nn.Conv2d(256, 256, kernel_size=3, padding=1), torch.nn.ReLU(inplace=True), torch.nn.BatchNorm2d(256),
                torch.nn.MaxPool2d(kernel_size=2, stride=2), torch.nn.Dropout(0.4),
                torch.nn.Conv2d(256, 512, kernel_size=3, padding=1), torch.nn.ReLU(inplace=True), torch.nn.BatchNorm2d(512),
                torch.nn.Conv2d(512, 512, kernel_size=3, padding=1), torch.nn.ReLU(inplace=True), torch.nn.BatchNorm2d(512),
                torch.nn.MaxPool2d(kernel_size=2, stride=2), torch.nn.Dropout(0.4),
            )
            self.classifier = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d((1, 1)), torch.nn.Flatten(),
                torch.nn.Linear(512, 256), torch.nn.ReLU(inplace=True), torch.nn.Dropout(0.5),
                torch.nn.Linear(256, 1)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    model = CustomCNN().to(device)
    model_path = "CustomCNN_best.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Please ensure it exists.")
    
    # Load the model weights with CPU mapping
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Perform prediction using a given model
def predict_with_model(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor).squeeze()
        probability = torch.sigmoid(output).item()
        prediction = "Real" if probability > 0.5 else "Fake"
        confidence = probability if prediction == "Real" else 1 - probability
        return prediction, confidence