from fastapi import APIRouter, HTTPException
from ..models.schemas import ImageRequest, PredictionResponse, FlagRequest
from ..utils.image_processing import decode_base64_image, extract_face, apply_filters, preprocess_image
from ..utils.model_utils import load_xception71_model, load_custom_cnn_model, predict_with_model
import torch
import logging

router = APIRouter()

logger = logging.getLogger(__name__)

xception71_model = load_xception71_model()
custom_cnn_model = load_custom_cnn_model()

@router.post("/predict", response_model=PredictionResponse)
async def predict_image(request: ImageRequest):
    try:
        logger.info(f"Received prediction request: model={request.model}, filters={request.filters}, isCameraInput={request.isCameraInput}")

        
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
            return {
                "isReal": True,
                "probability": 100.0,
                "model": request.model
            }

        # Predict with selected model
        if request.model == "CustomCNN":
            logger.info("Using CustomCNN model for prediction")
            prediction, confidence = predict_with_model(custom_cnn_model, processed_image)
            used_model = "CustomCNN"
        elif request.model == "Xception71":
            logger.info("Using Xception71 model for prediction")
            prediction, confidence = predict_with_model(xception71_model, processed_image)
            used_model = "Xception71"
        else:
            logger.error(f"Invalid model specified: {request.model}")
            raise ValueError(f"Unsupported model: {request.model}. Use 'CustomCNN' or 'Xception71'.")

        logger.info(f"Prediction completed: {prediction}, Confidence: {confidence}, Model: {used_model}")

        return {
            "isReal": prediction == "Real",
            "probability": confidence * 100,
            "model": used_model
        }

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/flag")
async def flag_prediction(request: FlagRequest):
    logger.info(f"Flagged prediction: {request}")
    return {"status": "success", "message": "Feedback recorded"}