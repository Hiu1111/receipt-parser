FROM python:3.12-slim

# Tesseract is a system binary, not a pip package. The health endpoint
# checks for it at runtime so a build that loses this line fails visibly
# instead of returning empty parses.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies before source, so editing code does not invalidate the
# layer that installs them.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY receipt_parser/ ./receipt_parser/

RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "receipt_parser.api:app", "--host", "0.0.0.0", "--port", "8000"]
