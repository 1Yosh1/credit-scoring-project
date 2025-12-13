# 1. Start with a lightweight Python base image
FROM python:3.9-slim

# 2. Set the working directory
WORKDIR /code

# --- NEW STEP: Install System Dependencies ---
# This fixes the "exit code 1" by giving Python the tools to compile libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 3. Copy requirements
COPY ./requirements.txt /code/requirements.txt

# 4. Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 5. Copy your source code and models
COPY ./src /code/src
COPY ./models /code/models

# 6. Run the app
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "80"]