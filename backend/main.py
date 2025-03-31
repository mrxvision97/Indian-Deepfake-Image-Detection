import base64
import io
from PIL import Image, ImageEnhance
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from ultralytics import YOLO
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://indian-deepfake-image-detection-jyqo.vercel.app",  # Vercel frontend domain
        "http://localhost:5173",  # For local development (Vite default port)
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


yolo_model_path = os.path.join(os.path.dirname(__file__), "yolov11m-face.pt")
yolo_model = YOLO(yolo_model_path)

def decode_base64_image(base64_string: str) -> Image.Image:
    try:
        if 'base64,' in base64_string:
            base64_string = base64_string.split('base64,')[1]
        
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {str(e)}")

def extract_face(image: Image.Image) -> Image.Image:
    try:
        # Convert PIL Image to numpy array (RGB)
        image_np = np.array(image)
        # Convert RGB to BGR for YOLO (OpenCV format)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Run YOLO face detection
        results = yolo_model(image_bgr)
        
        # Extract the first detected face
        if results and len(results[0].boxes) > 0:
            box = results[0].boxes[0]  # First face
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            face_np = image_np[y1:y2, x1:x2]  # Crop face from original RGB image
            
            # Ensure face is not empty
            if face_np.size == 0:
                raise ValueError("Extracted face region is empty")
            
            # Convert back to PIL Image
            face_image = Image.fromarray(face_np)
            return face_image
        else:
            raise ValueError("No face detected in the image")
    except Exception as e:
        raise ValueError(f"Face detection failed: {str(e)}")

def apply_filters(image: Image.Image, filters: dict) -> Image.Image:
    try:
        if filters.get('brightness', 100) != 100:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(filters['brightness'] / 100)
        
        if filters.get('contrast', 100) != 100:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(filters['contrast'] / 100)
        
        if filters.get('saturation', 100) != 100:
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(filters['saturation'] / 100)
        
        return image
    except Exception as e:
        raise ValueError(f"Failed to apply filters: {str(e)}")

def preprocess_image(image: Image.Image) -> np.ndarray:
    try:
        # Define validation transformations
        valid_transforms = A.Compose([
            A.Resize(height=224, width=224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        image = np.array(image)
        augmented = valid_transforms(image=image)
        image_tensor = augmented['image']
        # Add batch dimension (1, C, H, W)
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor
    except Exception as e:
        raise ValueError(f"Failed to preprocess image: {str(e)}")

# Add a health check endpoint for debugging
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Backend is running"}
