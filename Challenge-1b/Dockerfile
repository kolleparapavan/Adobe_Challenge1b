# Use an official Python runtime as the base image
FROM python:3.11-slim

# Set working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the script into the container
COPY challb.py .

# Create directories for input and output (will be overridden by volumes)
RUN mkdir -p /app/PDFs /app/analysis_output

# Set environment variables (optional, for better logging output)
ENV PYTHONUNBUFFERED=1

# Define the entry point (run the script)
CMD ["python", "challb.py"]