# -------------------------------------------------------------------
# Gemma-3n Vision-Language INT4 – Conversion Container
#
# • CPU-only base to avoid CUDA driver headaches; conversion is I/O-bound.
# • Pinned library versions known to expose ORTQuantizationConfig.
# • Produces ONNX-INT4 artefacts in /outputs (mount a host volume here).
# -------------------------------------------------------------------
FROM python:3.10-slim

ENV POETRY_VERSION=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# System deps – git (HF download fallback), tini for clean PID 1, basic build toolchain
RUN apt-get update && \
    apt-get install -y --no-install-recommends git build-essential curl tini && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements_src.txt

# -------------------------------------------------------------------
# Pin exact versions compatible with optimum 1.19 / ORTQuantizationConfig
# -------------------------------------------------------------------
RUN python -m pip install --upgrade pip && \
    # lightweight CPU wheel
    pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.1.2+cpu && \
    pip install \
        transformers==4.40.2 \
        optimum[onnxruntime]==1.19.2 \
        onnx==1.18.0 \
        sentencepiece==0.2.0 \
        safetensors==0.5.3 \
        accelerate==0.28.0 \
        # dev server optional
        fastapi==0.115.14 uvicorn[standard]==0.35.0

# -------------------------------------------------------------------
# Copy project
# -------------------------------------------------------------------
COPY . /workspace

# Default output directory inside container
ENV OUTPUT_DIR=/outputs
RUN mkdir -p ${OUTPUT_DIR}

# Entry point: run conversion (model ID can be overridden at runtime)
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "convert_gemma3n_vision_int4.py", "--output-dir", "/outputs"]