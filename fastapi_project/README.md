# AI Text Summarizer (FastAPI + PyTorch)

## Project Overview
This project provides a web interface for summarizing long texts using the BART model from Hugging Face Transformers. 
It uses FastAPI for the backend and pure HTML/CSS/JS for the frontend.

## Key Features
- **FastAPI Backend**: Efficient and modern Python web framework.
- **PyTorch & Transformers**: Uses `facebook/bart-large-cnn` for state-of-the-art summarization.
- **Long Text Support**: Implements a chunking strategy to handle texts longer than the model's limit (up to ~5000 tokens).
- **Dockerized**: Easy deployment using Docker and Docker Compose.

## Structure
- `app/main.py`: The entry point of the FastAPI application.
- `app/summarizer.py`: Contains the `TextSummarizer` class which handles model loading and the chunking logic.
- `app/templates/`: HTML templates.
- `app/static/`: CSS and JavaScript files.
- `Dockerfile` & `docker-compose.yml`: For containerization.

## How to Run

### Local Development (without Docker)
1. Navigate to this directory:
   ```bash
   cd fastapi_project
   ```
2. Create virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the server:
   ```bash
   uvicorn app.main:app --reload
   ```
4. Open http://localhost:8000 in your browser.

### Using Docker
1. Navigate to this directory:
   ```bash
   cd fastapi_project
   ```
2. Build and run:
   ```bash
   docker-compose up --build
   ```
3. Open http://localhost:8000 in your browser.

## Notes for AI
The summarization logic in `app/summarizer.py` splits long text into chunks of 3000 characters. Each chunk is summarized individually, and then the summaries are concatenated. This simple strategy effectively bypasses the 1024 token input limit of BART.
