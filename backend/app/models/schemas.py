from pydantic import BaseModel
from typing import Optional, Dict

class ImageFilters(BaseModel):
    brightness: int = 100
    contrast: int = 100
    saturation: int = 100

class ImageRequest(BaseModel):
    image: str
    model: str
    filters: Optional[ImageFilters] = None
    isCameraInput: Optional[bool] = False  # New field

class PredictionResponse(BaseModel):
    isReal: bool
    probability: float
    model: str

class FlagRequest(BaseModel):
    id: str
    feedback: str