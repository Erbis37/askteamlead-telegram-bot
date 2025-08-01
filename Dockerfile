FROM python:3.11-slim

WORKDIR /app

# Copy and install requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create download directory
RUN mkdir -p downloads

# Run as non-root user for better security
RUN useradd -m appuser
USER appuser

# Command to run the application
CMD ["python", "main.py"]
