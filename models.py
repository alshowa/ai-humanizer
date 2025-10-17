from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum

class HumanizePreset(str, Enum):
    CASUAL = "casual"
    PROFESSIONAL = "professional"
    STORYTELLING = "storytelling"
    ACADEMIC = "academic"
    CONVERSATIONAL = "conversational"
    PERSUASIVE = "persuasive"

class HumanizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100000)
    preset: HumanizePreset = HumanizePreset.CASUAL
    advanced_params: Optional[Dict[str, Any]] = Field(default_factory=dict)

class HumanizeResponse(BaseModel):
    job_id: str
    status: str
    original_text: Optional[str] = None
    humanized_text: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = Field(default_factory=dict)
    processing_time: Optional[float] = None
    error: Optional[str] = None