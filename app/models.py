from pydantic import BaseModel
from typing import Optional, Dict, Any
from enum import Enum

class HumanizePreset(str, Enum):
    CASUAL = "casual"
    PROFESSIONAL = "professional"
    STORYTELLING = "storytelling"

class HumanizeRequest(BaseModel):
    text: str
    preset: HumanizePreset = HumanizePreset.CASUAL
    advanced_params: Optional[Dict[str, Any]] = None

class HumanizeResponse(BaseModel):
    job_id: str
    status: str
    original_text: Optional[str] = None
    humanized_text: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None