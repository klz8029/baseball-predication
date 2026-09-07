# Serving image for the win-probability API.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# deps first for layer caching
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir "numpy" "pandas" "requests" "scikit-learn" \
    "cryptography" "fastapi" "uvicorn[standard]"

# model + serving code (flat module layout)
COPY montecarlo.py blend.py metrics.py ./
COPY serving ./serving
COPY data/feature_cache.pkl ./data/feature_cache.pkl

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
