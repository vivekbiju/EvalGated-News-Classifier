FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py .

# Expose Hugging Face's default Space port
EXPOSE 7860

# Run Uvicorn explicitly bound to 0.0.0.0 and port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]