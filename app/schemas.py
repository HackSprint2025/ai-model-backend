from pydantic import BaseModel, conlist, Field
from typing import List


class PredictionRequest(BaseModel):
    features: conlist(float, min_length=24, max_length=24) = Field(
        ..., description="List of 24 float features in the correct order"
    )

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float = None 
