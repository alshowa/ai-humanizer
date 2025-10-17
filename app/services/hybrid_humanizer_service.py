import spacy
import nltk
from textblob import TextBlob
from transformers import pipeline
import time
import re
import random
from typing import Dict, Any

class HybridHumanizerService:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        try:
            self.paraphraser = pipeline("text2text-generation", model="t5-small")
        except:
            self.paraphraser = None
    
    async def humanize(self, text: str, style: str = "casual") -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            # Simple humanization logic
            humanized = self.simple_humanize(text, style)
            
            return {
                "job_id": f"job_{int(time.time())}",
                "status": "completed",
                "original_text": text,
                "humanized_text": humanized,
                "metrics": {
                    "processing_time": time.time() - start_time,
                    "provider": "simple_nlp"
                }
            }
        except Exception as e:
            return {
                "job_id": f"job_{int(time.time())}",
                "status": "failed",
                "error": str(e)
            }
    
    def simple_humanize(self, text: str, style: str) -> str:
        """Simple text humanization"""
        transformations = {
            "cannot": "can't",
            "will not": "won't",
            "do not": "don't", 
            "does not": "doesn't",
            "is not": "isn't",
            "are not": "aren't",
            "utilize": "use",
            "commence": "start",
            "approximately": "about",
            "assistance": "help",
            "demonstrate": "show",
            "facilitate": "help"
        }
        
        result = text
        for formal, casual in transformations.items():
            result = result.replace(formal, casual)
        
        # Try transformer paraphrasing if available
        if self.paraphraser:
            try:
                paraphrased = self.paraphraser(
                    f"paraphrase: {result}",
                    max_length=len(result) + 100,
                    num_return_sequences=1
                )[0]['generated_text']
                result = paraphrased
            except:
                pass
        
        return result