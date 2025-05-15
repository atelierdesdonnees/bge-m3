ARG CUDA_VERSION
# FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu22.04
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

# Set non-interactive frontend to avoid tzdata configuration prompt
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.12 and necessary tools
RUN apt-get update && \
    apt-get install -y \
    git \
    wget \
    python3 \
    python3-pip

# Import HF token
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
    export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    fi

WORKDIR /temp

# Create and set permissions for /root directory (Podman compatibility)
RUN mkdir -p /root && chmod 750 /root

# Copy setup files
COPY . .

# Run setup script with Python 3
RUN python3 setup_environment.py \
    --cuda-version ${CUDA_VERSION} \
    --src-dir ./src \
    --env-file ./environment.env \
    --requirements-file ./additional-requirements.txt && \
    rm -rf /temp

WORKDIR /worker-infinity-embedding

# Charger les variables d'environnement au démarrage
CMD ["/bin/bash", "-c", "if [ -f /root/.env ]; then export $(cat /root/.env | grep -v '^#' | xargs); fi && python3 src/handler.py"]
