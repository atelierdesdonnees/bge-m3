#!/bin/bash
set -e # Exit on error

# First update and install python3-apt for apt_pkg
apt-get update

# Install Python 3.12 and dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget

# Check if CUDA_VERSION is set
if [ -z "${CUDA_VERSION}" ]; then
    if command -v nvidia-smi &>/dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
        echo "Detected CUDA version: ${CUDA_VERSION}"
    else
        echo "Error: CUDA_VERSION not set and nvidia-smi not available"
        exit 1
    fi
fi

# Run setup script with core requirements
python3 setup_environment.py \
    --cuda-version "${CUDA_VERSION}" \
    --src-dir ./src \
    --env-file ./environment.env \
    --requirements-file ./requirements.txt
