FROM python:3.14-slim

WORKDIR /app

# Copy requirements
COPY chat-interface/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY chat-interface/app.py .
COPY llmfs_lib.py .
COPY gpt2-medium-355M-sft-it.pth .

# Create public directory for static assets
RUN mkdir -p public

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CHAINLIT_HOST=0.0.0.0
ENV CHAINLIT_PORT=8000

# Expose port
EXPOSE 8000

# Default command
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]