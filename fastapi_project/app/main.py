import sys
import os

# Add project root to sys.path to resolve 'app' module
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from app.summarizer import TextSummarizer
import uvicorn

app = FastAPI(title="Text Summarization API")

# Mount static files using absolute paths
static_dir = os.path.join(base_dir, "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Setup templates using absolute paths
templates_dir = os.path.join(base_dir, "templates")
templates = Jinja2Templates(directory=templates_dir)

# Initialize model (Lazy loading or on startup)
# We initialize on a global variable to keep it in memory
summarizer_model = None

@app.on_event("startup")
async def startup_event():
    global summarizer_model
    # This might take time on startup
    summarizer_model = TextSummarizer()

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/summarize")
async def summarize_text(text: str = Form(...)):
    if not text:
        return {"summary": "Please provide text."}
    
    try:
        # Process text - assuming text could be up to 5000 tokens as requested
        summary = summarizer_model.summarize(text, max_length=200, min_length=80)
        return {"summary": summary}
    except Exception as e:
        return {"summary": f"Error during summarization: {str(e)}"}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
