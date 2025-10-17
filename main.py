from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Dict, List, Optional
import time

# Simple import from our app structure
from app.services.hybrid_humanizer_service import HybridHumanizerService

app = FastAPI(title="AI Humanizer", version="1.0.0")
templates = Jinja2Templates(directory="templates")
humanizer = HybridHumanizerService()


STYLES: List[Dict[str, str]] = [
    {
        "id": "casual",
        "name": "Casual",
        "description": "Relaxed, friendly voice",
        "icon": "😊",
        "best_for": "Social updates & quick messages",
    },
    {
        "id": "professional",
        "name": "Professional",
        "description": "Clear and confident business tone",
        "icon": "💼",
        "best_for": "Emails, proposals & formal docs",
    },
    {
        "id": "storytelling",
        "name": "Storytelling",
        "description": "Engaging narrative flair",
        "icon": "📚",
        "best_for": "Blogs, stories & presentations",
    },
    {
        "id": "academic",
        "name": "Academic",
        "description": "Scholarly and precise language",
        "icon": "🎓",
        "best_for": "Research papers & reports",
    },
]


EXAMPLES: List[Dict[str, str]] = [
    {
        "id": "marketing",
        "description": "Marketing blurb",
        "original": "Our proprietary solution leverages a robust suite of capabilities to expedite cross-functional synergy across the organization.",
        "style": "casual",
    },
    {
        "id": "status_update",
        "description": "Team status update",
        "original": "I would like to inform you that we have finalized the integration module and will proceed with deployment upon stakeholder approval.",
        "style": "professional",
    },
    {
        "id": "education",
        "description": "Educational paragraph",
        "original": "Photosynthesis is the process whereby green plants utilize chlorophyll to convert light energy into chemical energy in the form of glucose.",
        "style": "storytelling",
    },
]


class HumanizeRequest(BaseModel):
    text: str
    style: str = "casual"
    enhance_readability: bool = True
    use_contractions: bool = True
    vary_sentences: bool = True


class HumanizeResponse(BaseModel):
    success: bool
    message: str
    original_text: str
    humanized_text: str
    style_used: str
    processing_time: float
    readability_improvement: float
    word_count: int
    metrics: Dict[str, float]
    style_details: Optional[Dict[str, str]]


def _simple_readability_score(text: str) -> float:
    """Very small heuristic to approximate readability for demo purposes."""
    if not text:
        return 0.0

    words = [word for word in text.replace("\n", " ").split(" ") if word]
    sentences = text.count(".") + text.count("!") + text.count("?")
    sentences = sentences or 1

    avg_sentence_length = len(words) / sentences
    avg_word_length = sum(len(word.strip(".,!?;:")) for word in words) / len(words)

    # Produce a bounded score between 0 and 100 where higher is easier to read
    score = 100 - (avg_sentence_length * 2 + avg_word_length * 5)
    return max(0.0, min(score, 100.0))


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/v1/styles")
async def get_styles():
    return {"styles": STYLES}


@app.get("/api/v1/examples")
async def get_examples():
    return {"examples": EXAMPLES}


@app.post("/api/v1/humanize", response_model=HumanizeResponse)
async def humanize_text(request: HumanizeRequest):
    start_time = time.time()

    try:
        result = await humanizer.humanize(request.text, request.style)

        if result.get("status") != "completed":
            raise HTTPException(500, result.get("error", "Unknown error"))

        original_text = result["original_text"]
        humanized_text = result["humanized_text"]

        original_readability = _simple_readability_score(original_text)
        humanized_readability = _simple_readability_score(humanized_text)
        readability_improvement = humanized_readability - original_readability

        processing_time = result.get("metrics", {}).get("processing_time", time.time() - start_time)
        word_count = len([word for word in humanized_text.replace("\n", " ").split(" ") if word])

        style_details = next((style for style in STYLES if style["id"] == request.style), None)

        return HumanizeResponse(
            success=True,
            message="Text humanized successfully.",
            original_text=original_text,
            humanized_text=humanized_text,
            style_used=request.style,
            processing_time=processing_time,
            readability_improvement=readability_improvement,
            word_count=word_count,
            metrics={
                "processing_time": processing_time,
                "readability_before": original_readability,
                "readability_after": humanized_readability,
                "enhance_readability": float(request.enhance_readability),
                "use_contractions": float(request.use_contractions),
                "vary_sentences": float(request.vary_sentences),
            },
            style_details=style_details,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Humanization failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)