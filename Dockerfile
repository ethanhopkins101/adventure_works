# 1. Use a stable Python version
FROM python:3.11-slim

# 2. Install system-level dependencies (required for Prophet/PyMC/C++ builds)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 3. Set working directory
WORKDIR /app

# 4. Copy and install dependencies first (optimizes build cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the project code
COPY . .

# 6. Default command (Can be overridden by the scheduler)
CMD ["python", "main.py"]