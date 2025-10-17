from app.services.hybrid_humanizer import HybridTextHumanizer
from app.models import HumanizeRequest, HumanizeResponse
import time
import asyncio

class HybridHumanizerService:
    def __init__(self, use_gpu: bool = False):
        self.humanizer = HybridTextHumanizer(use_gpu=use_gpu)
    
    async def humanize(self, request: HumanizeRequest) -> HumanizeResponse:
        start_time = time.time()
        
        try:
            # Run humanization in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self.humanizer.humanize_text, 
                request.text, 
                request.preset.value
            )
            
            processing_time = time.time() - start_time
            
            return HumanizeResponse(
                job_id=f"hybrid_{int(start_time)}",
                status="completed",
                original_text=result["original_text"],
                humanized_text=result["humanized_text"],
                metrics={
                    "processing_time": processing_time,
                    "provider": "hybrid_transformer_nlp",
                    **result["metrics"]
                }
            )
            
        except Exception as e:
            return HumanizeResponse(
                job_id=f"hybrid_{int(start_time)}",
                status="failed",
                error=str(e)
            )