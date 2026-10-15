#!/bin/bash
# Set up (first call) and activate (every call) the Python environment on DTU HPC.
#
#   source init.sh
#
# Settings can be overridden with environment variables, e.g.
#   PY_MODULE=python3/3.12.4 TORCH_CUDA=cu126 source init.sh
#
# Check which Python modules exist with:  module avail python3
# Check the GPU driver / max. CUDA version on a GPU node (e.g. after `a100sh`) with:  nvidia-smi

TORCH_CUDA=${TORCH_CUDA:-cu126}
VENV_DIR=${VENV_DIR:-$HOME/venvs/facts}
# Use the Python version the venv was created with (read from pyvenv.cfg), else the default
if [ -z "$PY_MODULE" ] && [ -f "$VENV_DIR/pyvenv.cfg" ]; then
    PY_VERSION=$(grep -E '^version' "$VENV_DIR/pyvenv.cfg" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    [ -n "$PY_VERSION" ] && PY_MODULE="python3/$PY_VERSION"
fi
PY_MODULE=${PY_MODULE:-python3/3.12.11}
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! module load "$PY_MODULE" 2>/dev/null; then
    echo "Could not load module '$PY_MODULE'. Available Python modules:"
    module avail python3 2>&1 | grep -o 'python3/[0-9.]*' | sort -uV
    echo "Re-run with e.g.: PY_MODULE=python3/<version> source init.sh"
    return 1 2>/dev/null || exit 1
fi

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "Creating virtual environment in $VENV_DIR (Python: $(python3 --version))"
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install torch torchvision --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
    pip install -r "$REPO_DIR/requirements.txt"
    # Download the LPIPS (AlexNet) weights now, in case compute nodes have no internet access
    python -c "import lpips; lpips.LPIPS(net='alex', verbose=False)"
else
    source "$VENV_DIR/bin/activate"
fi

python -c "import torch; print(f'torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}')"
