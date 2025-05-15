#!/bin/bash
set -e

# Function to show usage
show_usage() {
    echo "Usage: $0 [--tag <image-tag>]"
    echo "  --tag : Specify the image tag (default: infinity-worker:latest)"
    exit 1
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect container engine
detect_container_engine() {
    if command_exists podman; then
        echo "podman"
    elif command_exists docker; then
        echo "docker"
    else
        echo "none"
    fi
}

# Function to print installation instructions
print_install_instructions() {
    echo "No container engine found. You need to install either Podman or Docker."
    echo ""
    echo "To install Podman:"
    case "$(uname -s)" in
    Linux*)
        echo "  Ubuntu/Debian: sudo apt-get install -y podman"
        echo "  Fedora: sudo dnf install -y podman"
        echo "  Other distributions: Visit https://podman.io/getting-started/installation"
        ;;
    Darwin*)
        echo "  macOS: brew install podman"
        ;;
    esac
    echo ""
    echo "To install Docker:"
    echo "  Visit https://docs.docker.com/get-docker/"
    exit 1
}

# Parse command line arguments
IMAGE_TAG="infinity-worker:latest"
while [[ "$#" -gt 0 ]]; do
    case $1 in
    --tag)
        IMAGE_TAG="$2"
        shift
        ;;
    --help) show_usage ;;
    *)
        echo "Unknown parameter: $1"
        show_usage
        ;;
    esac
    shift
done

# Check if environment.env exists
if [ ! -f environment.env ]; then
    echo "Error: environment.env file not found"
    exit 1
fi

# Extract CUDA version from environment.env
CUDA_VERSION=$(grep CUDA_VERSION environment.env | cut -d= -f2)

if [ -z "$CUDA_VERSION" ]; then
    echo "Error: CUDA_VERSION not found in environment.env"
    exit 1
fi

# Detect container engine
CONTAINER_ENGINE=$(detect_container_engine)

if [ "$CONTAINER_ENGINE" = "none" ]; then
    echo "Error: No container engine found."
    print_install_instructions
    exit 1
fi

# Extract the HF token if it exists in the environment
HF_TOKEN_ARG=""
if [ -n "$HF_TOKEN" ]; then
    HF_TOKEN_ARG="--secret id=HF_TOKEN,env=HF_TOKEN"
fi

# Build the container image
echo "Building using $CONTAINER_ENGINE with CUDA version: $CUDA_VERSION"
echo "Image will be tagged as: $IMAGE_TAG"

if [ "$CONTAINER_ENGINE" = "podman" ]; then
    # Podman-specific build command
    BUILDAH_LAYERS=true podman build \
        --format docker \
        --platform=linux/amd64 \
        --build-arg CUDA_VERSION="$CUDA_VERSION" \
        $HF_TOKEN_ARG \
        -t "$IMAGE_TAG" \
        .
else
    # Docker build command
    DOCKER_BUILDKIT=1 docker build \
        --build-arg CUDA_VERSION="$CUDA_VERSION" \
        $HF_TOKEN_ARG \
        -t "$IMAGE_TAG" \
        .
fi

echo "Build completed successfully using $CONTAINER_ENGINE!"
echo "Image tagged as: $IMAGE_TAG"
