from typing import Literal
from pydantic import BaseModel, Field


class PatientInput(BaseModel):
    sexe: Literal["F", "H"]
    age: float = Field(..., ge=0, le=120)
    zone_vie: Literal["U", "R"]
    source: Literal["appel", "chat"]
    freq_cardiaque: float = Field(..., ge=30, le=250,
                                  description="Fréquence cardiaque en bpm")
    tension_sys: float = Field(..., ge=50, le=300,
                                description="Tension systolique en mmHg")
    temp: float = Field(..., ge=34.0, le=43.0,
                         description="Température corporelle en °C")
    sat_oxygene: float = Field(..., ge=50.0, le=100.0,
                                description="Saturation O2 en %")
    antecedents: Literal[0, 1]
    duree_symptomes: float = Field(..., ge=0,
                                    description="Durée des symptômes en heures")
    description_symptomes: str = Field(..., min_length=3,
                                        description="Description libre des symptômes")


class PredictionResponse(BaseModel):
    niveau_urgence: int
    label: str
    probabilites: dict[str, float]
    timestamp: str
    duration_ms: float
    model_name: str
    threshold_class_2: float

class RetrainResponse(BaseModel):
    status: str
    model_updated: bool
    new_metrics: dict[str, float]
    current_metrics: dict[str, float]
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    version: str
    model: str
    timestamp: str