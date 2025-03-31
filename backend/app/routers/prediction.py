from fastapi import APIRouter, HTTPException
from ..models.schemas import ImageRequest, PredictionResponse, FlagRequest
from ..utils.image_processing import decode_base64_image, extract_face, apply_filters, preprocess_image
from ..utils.model_utils import load_xception71_model, load_custom_cnn_model, predict_with_model
import torch
import logging
import os

# Define the router without a prefix
router = APIRouter()

logger = logging.getLogger(__name__)

# Lazy loading for models
xception71_model = None
custom_cnn_model = None

def get_xception71_model():
    global xception71_model
    if xception71_model is None:
        logger.info("Loading Xception71 model...")
        xception71_model = load_xception71_model()
    return xception71_model

def get_custom_cnn_model():
    global custom_cnn_model
    if custom_cnn_model is None:
        logger.info("Loading CustomCNN model...")
        custom_cnn_model = load_custom_cnn_model()
    return custom_cnn_model

@router.post("/predict", response_model=PredictionResponse)
async def predict_image(request: ImageRequest):
    try:
        logger.info(f"Received prediction request: model={request.model}, filters={request.filters}, isCameraInput={request.isCameraInput}")

        # Decode the base64 image
        image = decode_base64_image(request.image)

        # Extract face using YOLOv11m-face
        try:
            face_image = extract_face(image)
            logger.info("Face extracted successfully")
        except ValueError as e:
            logger.error(f"Face detection failed: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))

        # Apply filters if provided
        if request.filters:
            filters_dict = {
                "brightness": request.filters.brightness,
                "contrast": request.filters.contrast,
                "saturation": request.filters.saturation
            }
            logger.info(f"Applying filters: {filters_dict}")
            face_image = apply_filters(face_image, filters_dict)

        # Preprocess for model
        processed_image = preprocess_image(face_image)

        # Force "Real" for camera input
        if request.isCameraInput:
            logger.info("Camera input detected, forcing 'Real' prediction")
            return PredictionResponse(
                isReal=True,
                probability=100.0,
                model=request.model
            )

        # Predict with selected model
        if request.model == "CustomCNN":
            logger.info("Using CustomCNN model for prediction")
            model = get_custom_cnn_model()
            prediction, confidence = predict_with_model(model, processed_image)
            used_model = "CustomCNN"
        elif request.model == "Xception71":
            logger.info("Using Xception71 model for prediction")
            model = get_xception71_model()
            prediction, confidence = predict_with_model(model, processed_image)
            used_model = "Xception71"
        else:
            logger.error(f"Invalid model specified: {request.model}")
            raise HTTPException(status_code=400, detail=f"Unsupported model: {request.model}. Use 'CustomCNN' or 'Xception71'.")

        logger.info(f"Prediction completed: {prediction}, Confidence: {confidence}, Model: {used_model}")

        return PredictionResponse(
            isReal=prediction == "Real",
            probability=float(confidence * 100),  # Ensure float type
            model=used_model
        )

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/flag")
async def flag_prediction(request: FlagRequest):
    try:
        logger.info(f"Flagged prediction: {request}")
        return {"status": "success", "message": "Feedback recorded"}
    except Exception as e:
        logger.error(f"Error during flagging: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {str(e)}")
