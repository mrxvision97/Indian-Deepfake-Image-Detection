import base64
import io
from PIL import Image, ImageEnhance
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from ultralytics import YOLO

yolo_model = YOLO(r"C:\Users\am998\OneDrive\Desktop\Projets UI\Indian Deepfakes Updated\backend\yolov11m-face.pt")

def decode_base64_image(base64_string: str) -> Image.Image:
    
    if 'base64,' in base64_string:
        base64_string = base64_string.split('base64,')[1]
    
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image

def extract_face(image: Image.Image) -> Image.Image:
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

def apply_filters(image: Image.Image, filters: dict) -> Image.Image:
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

def preprocess_image(image: Image.Image) -> np.ndarray:
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