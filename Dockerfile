# BERT phishing classifier — runtime image.
# Base on pytorch:1.3-cuda10.1 to match the 2019 toolchain.
FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/cache

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY README.md ./

RUN mkdir -p /cache && chmod -R a+rw /cache

# Default: serve a CLI entrypoint for inference
ENTRYPOINT ["python", "-m", "src.predict"]
CMD ["--help"]
