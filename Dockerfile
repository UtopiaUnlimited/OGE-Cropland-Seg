# Python plus official CUDA-enabled PyTorch wheels keeps the submission runtime small.
ARG BASE_IMAGE=python:3.11-slim-bookworm@sha256:90d117a7f1ef6535dc75831e1a9184725757fede0d4d424d7977fb5429d50610
FROM ${BASE_IMAGE}
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HOME=/tmp/huggingface \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
COPY docker/requirements-inference.txt /tmp/requirements-inference.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-inference.txt \
    && python -m pip check \
    && python -c "import torch; assert torch.__version__.split('+')[0] == '2.6.0'; assert torch.version.cuda == '12.4'"
# Rasterio/GDAL wheels require the system Expat XML library.
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 && rm -rf /var/lib/apt/lists/*
COPY cropland/ /app/cropland/
COPY submission/build/best.pt /opt/oge/best.pt
COPY submission/build/manifest.json /opt/oge/manifest.json
RUN python -c "import hashlib,json; p='/opt/oge/'; m=json.load(open(p+'manifest.json')); f=open(p+'best.pt','rb'); assert hashlib.file_digest(f,'sha256').hexdigest()==m['checkpoint_sha256']" \
    && python -m cropland.inference --help
LABEL org.opencontainers.image.title="OGE Cropland SegFormer-B5" \
      org.opencontainers.image.source="https://github.com/UtopiaUnlimited/OGE-Cropland-Seg"
# OGE only appends --input_dataset and --output_dir; weights are inside the image.
ENTRYPOINT ["python", "-m", "cropland.inference", "--pretrained", "/opt/oge/best.pt", "--device", "cuda"]
