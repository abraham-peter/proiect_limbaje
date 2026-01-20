FROM python:3.10-slim

WORKDIR /app

# Install system dependencies if needed (e.g., build-essential)
# apt-get update && apt-get install -y build-essential

COPY requirements.txt .

# Install dependencies
# Using --no-cache-dir to keep image small
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (FastAPI default is 8000)
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "localhost", "--port", "8000"]
