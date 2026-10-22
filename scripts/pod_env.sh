# Shared settings for the pod scripts (source it, don't run it).
# Caches live on the network volume, so a new pod reuses Python, wheels and model
# weights instead of downloading them again.

export WORKSPACE="${WORKSPACE:-/workspace}"
export REPO_DIR="$WORKSPACE/pitchstate"
export LOG_DIR="$WORKSPACE/logs"

export UV_INSTALL_DIR="$WORKSPACE/.cache/uv-bin"
export UV_CACHE_DIR="$WORKSPACE/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$WORKSPACE/.cache/uv-python"
export UV_LINK_MODE=copy  # network volumes may not support hard links

export HF_HOME="$WORKSPACE/.cache/huggingface"    # SigLIP
export TORCH_HOME="$WORKSPACE/.cache/torch"       # PARSeq, torchvision weights
export YOLO_CONFIG_DIR="$WORKSPACE/.cache/ultralytics"

export PATH="$UV_INSTALL_DIR:$PATH"
