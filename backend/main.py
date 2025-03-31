from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import prediction
import logging
import uvicorn


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Deepfake Detection API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://indian-deepfake-image-detection-jyqo.vercel.app"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(prediction.router)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Deepfake Detection API...")
    try:
        # Log model initialization status
        from app.routers.prediction import xception71_model, custom_cnn_model
        logger.info("Models initialized successfully.")
    except Exception as e:
        logger.error(f"Error during model initialization: {str(e)}")
        raise RuntimeError("Failed to initialize models. Check logs for details.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Deepfake Detection API...")

if __name__ == "__main__":
    logger.info("Running the API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
