FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install packages without additional system deps
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --timeout 300 -r requirements.txt

# Copy application
COPY app/ app/
COPY models/ models/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
