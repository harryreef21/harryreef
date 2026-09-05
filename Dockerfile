# OCR backend — build once, deploy anywhere (local, Render, Railway, a VM, Cloud Run)
FROM python:3.11-slim

WORKDIR /app

# EasyOCR needs these system libs to decode images / run torch on CPU
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ocr_backend_fastapi.py .

# Pre-download the Thai+English EasyOCR models at build time so the
# first real request isn't slowed down by a model download.
RUN python -c "import easyocr; easyocr.Reader(['th','en'], gpu=False)"

EXPOSE 8000
CMD ["uvicorn", "ocr_backend_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
