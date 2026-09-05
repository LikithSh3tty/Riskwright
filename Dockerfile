# One image, three roles. The loader, the API, and the UI differ only by the
# command compose gives them, so the layer cache is built once and reused.
FROM python:3.11-slim

# libgomp1 is LightGBM's OpenMP runtime and is not in the slim base.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    DOCKER_CONTAINER=1

WORKDIR /app

# Dependencies first so code edits do not invalidate the install layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code is baked in rather than bind-mounted. The repository lives under a
# OneDrive path containing a space and a '!', and mounting it would drag
# sync locks into the container.
COPY src/ ./src/
COPY app/ ./app/
COPY ui/ ./ui/
COPY sql/ ./sql/
COPY configs/ ./configs/
COPY models/ ./models/
COPY requirements.txt ./

EXPOSE 8000 8501

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
