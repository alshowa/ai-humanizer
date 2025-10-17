from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import time

# Simple import from our app structure
from app.services.hybrid_humanizer_service import HybridHumanizerService

app = FastAPI(title="AI Humanizer", version="1.0.0")
humanizer = HybridHumanizerService()

class SimpleRequest(BaseModel):
    text: str
    style: str = "casual"

class SimpleResponse(BaseModel):
    success: bool
    original_text: str
    humanized_text: str
    processing_time: float

@app.post("/api/humanize", response_model=SimpleResponse)
async def humanize_text(request: SimpleRequest):
    start_time = time.time()
    
    try:
        result = await humanizer.humanize(request.text, request.style)
        
        if result["status"] == "completed":
            return SimpleResponse(
                success=True,
                original_text=result["original_text"],
                humanized_text=result["humanized_text"],
                processing_time=result["metrics"]["processing_time"]
            )
        else:
            raise HTTPException(500, result.get("error", "Unknown error"))
            
    except Exception as e:
        raise HTTPException(500, f"Humanization failed: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
    <head>
        <title>AI Humanizer</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-100 min-h-screen p-8">
        <div class="max-w-4xl mx-auto">
            <h1 class="text-4xl font-bold text-center text-blue-600 mb-8">AI Humanizer</h1>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div class="bg-white p-6 rounded-lg shadow">
                    <h2 class="text-2xl font-semibold mb-4">Input</h2>
                    <textarea id="inputText" class="w-full h-40 p-4 border rounded" 
                              placeholder="Paste AI text here...">The utilization of artificial intelligence represents significant advancement.</textarea>
                    
                    <select id="styleSelect" class="w-full p-2 border rounded mt-4">
                        <option value="casual">Casual</option>
                        <option value="professional">Professional</option>
                    </select>
                    
                    <button onclick="humanize()" class="w-full bg-blue-500 text-white p-3 rounded mt-4">
                        Humanize Text
                    </button>
                </div>
                
                <div class="bg-white p-6 rounded-lg shadow">
                    <h2 class="text-2xl font-semibold mb-4">Output</h2>
                    <div id="output" class="w-full h-40 p-4 border rounded bg-gray-50">
                        Humanized text will appear here...
                    </div>
                    <div id="loading" class="hidden text-blue-500 text-center mt-4">
                        Processing...
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            async function humanize() {
                const input = document.getElementById('inputText').value;
                const style = document.getElementById('styleSelect').value;
                const output = document.getElementById('output');
                const loading = document.getElementById('loading');
                
                loading.classList.remove('hidden');
                
                try {
                    const response = await fetch('/api/humanize', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({text: input, style: style})
                    });
                    
                    const result = await response.json();
                    output.textContent = result.humanized_text;
                } catch (error) {
                    output.textContent = 'Error: ' + error.message;
                } finally {
                    loading.classList.add('hidden');
                }
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)