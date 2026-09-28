# Serving image: numpy + ONNX Runtime + FastAPI only (no PyTorch, no SciPy).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PULMOEIT_MODEL_DIR=/app/models

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[serve]"

COPY models/pulmoeit.onnx models/qc.npz models/model_card.json ./models/

RUN useradd --create-home --uid 1000 app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "pulmoeit.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
