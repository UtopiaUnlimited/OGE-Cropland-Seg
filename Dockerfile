# Build on an Internet-connected host; the resulting image runs fully offline.
ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
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
COPY docker/requirements-inference.txt /tmp/requirements-inference.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-inference.txt \
    && python -m pip check \
    && python -c "import torch; assert torch.__version__.split('+')[0] == '2.6.0'"
COPY cropland/ /app/cropland/
COPY submission/build/best.pt /opt/oge/best.pt
COPY submission/build/manifest.json /opt/oge/manifest.json
RUN python -c "import hashlib,json; p='/opt/oge/'; m=json.load(open(p+'manifest.json')); f=open(p+'best.pt','rb'); assert hashlib.file_digest(f,'sha256').hexdigest()==m['checkpoint_sha256']" \
    && python -m cropland.inference --help
LABEL org.opencontainers.image.title="OGE Cropland SegFormer-B5" \
      org.opencontainers.image.source="https://github.com/UtopiaUnlimited/OGE-Cropland-Seg"
# OGE only appends --input_dataset and --output_dir; weights are inside the image.
ENTRYPOINT ["python", "-m", "cropland.inference", "--pretrained", "/opt/oge/best.pt", "--device", "cuda"]
