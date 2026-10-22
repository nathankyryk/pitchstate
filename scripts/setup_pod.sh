#!/usr/bin/env bash
# Set up a fresh Runpod pod that has the `pitchstate` network volume mounted at /workspace.
#
# First pod (repo not on the volume yet):
#   curl -fsSL https://raw.githubusercontent.com/nathankyryk/pitchstate/master/scripts/setup_pod.sh | bash
# Any later pod:
#   bash /workspace/pitchstate/scripts/setup_pod.sh
#
# Safe to re-run: every step checks what is already there. Python, packages, data and
# model weights are kept on the volume, so a second pod is ready in a minute or two.
# Ends with `pitchstate doctor`; a non-zero exit means something needs attention.

set -euo pipefail

REPO_URL="https://github.com/nathankyryk/pitchstate.git"
UV_VERSION="0.13.1"  # same uv that wrote uv.lock
WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="$WORKSPACE/pitchstate"
SAM_BYTES=3502755717  # size of facebook/sam3.1 sam3.1_multiplex.pt
PITCH_URL="https://drive.google.com/uc?id=1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf"  # Roboflow sports, pitch keypoints

mkdir -p "$WORKSPACE/logs"
LOG="$WORKSPACE/logs/setup_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

step() { printf '\n== %s ==\n' "$*"; }
fail() { printf '!! %s\n' "$*" >&2; exit 1; }

step "GPU and driver"
command -v nvidia-smi >/dev/null || fail "no nvidia-smi: this is not a GPU pod"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
cuda=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -n1)
echo "driver supports CUDA $cuda"
awk -v v="$cuda" 'BEGIN { exit !(v + 0 >= 12.8) }' ||
  fail "PyTorch here is built for CUDA 12.8; redeploy the pod with the CUDA 12.8+ filter"

step "System tools"
missing=()
for tool in git curl tmux; do command -v "$tool" >/dev/null || missing+=("$tool"); done
if ((${#missing[@]})); then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${missing[@]}"
fi
echo "git, curl and tmux present"

step "Repository"
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" pull --ff-only
else
  git clone "$REPO_URL" "$REPO_DIR"
fi
git -C "$REPO_DIR" log --oneline -1
# shellcheck source=scripts/pod_env.sh
source "$REPO_DIR/scripts/pod_env.sh"
cd "$REPO_DIR"

step "uv $UV_VERSION"
if ! command -v uv >/dev/null || [ "$(uv --version | cut -d' ' -f2)" != "$UV_VERSION" ]; then
  curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_NO_MODIFY_PATH=1 sh
fi
uv --version

step "Python environment (exactly as uv.lock)"
uv sync --frozen
uv run python - <<'EOF'
import torch
assert torch.cuda.is_available(), "PyTorch cannot see the GPU"
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}) on {torch.cuda.get_device_name(0)}")
EOF

step "Model caches on the volume"
# RF-DETR downloads its pretrained weights to ~/.roboflow; keep them on the volume.
mkdir -p "$WORKSPACE/.cache/roboflow"
if [ ! -L "$HOME/.roboflow" ]; then
  rm -rf "$HOME/.roboflow"
  ln -s "$WORKSPACE/.cache/roboflow" "$HOME/.roboflow"
fi
echo "$HOME/.roboflow -> $(readlink "$HOME/.roboflow")"

step "SoccerNet GSR data"
data="$REPO_DIR/data/SoccerNetGS/gamestate-2024"
have_data=1
for split in train valid test; do
  # The marker is written only after a whole split is unpacked, so an unpack cut short
  # (a stopped pod) is redone instead of trusted.
  [ -f "$data/$split/.extracted" ] || have_data=0
done
if [ "$have_data" = 0 ]; then
  uv run python scripts/download_gsr.py  # ~28 GB, a few minutes inside the datacenter
fi
for split in train valid test; do
  # Only a fully unpacked split makes its zip disposable: free the space.
  [ -f "$data/$split.zip" ] && [ -f "$data/$split/.extracted" ] && rm -f "$data/$split.zip"
  echo "$split: $(find "$data/$split" -mindepth 1 -maxdepth 1 -type d | wc -l) clips"
done

step "Model weights"
mkdir -p weights
if [ ! -f weights/football-pitch-detection.pt ]; then
  uvx gdown -q -O weights/football-pitch-detection.pt "$PITCH_URL" ||
    echo "!! pitch weights download failed: copy weights/football-pitch-detection.pt from the laptop"
fi
if [ -f weights/sam3.1_multiplex.pt ]; then
  size=$(stat -c %s weights/sam3.1_multiplex.pt)
  [ "$size" = "$SAM_BYTES" ] || fail "weights/sam3.1_multiplex.pt is $size bytes, expected $SAM_BYTES: copy it again"
else
  echo "!! SAM 3.1 weights missing (gated, so they come from the laptop). In PowerShell on the laptop:"
  printf '   scp -P <PORT> %s root@<POD_IP>:%s/weights/\n' \
    'C:\Users\natha\code\pitchstate\weights\sam3.1_multiplex.pt' "$REPO_DIR"
  echo "   (<POD_IP> and <PORT>: the pod's 'SSH over exposed TCP' line in the Runpod console)"
fi
ls -lh weights

step "pitchstate doctor"
if uv run pitchstate doctor; then
  printf '\nReady. Next: bash scripts/run_gpu_jobs.sh check   (log: %s)\n' "$LOG"
else
  printf '\nNot ready: fix the !! lines above, then re-run this script. (log: %s)\n' "$LOG"
  exit 1
fi
