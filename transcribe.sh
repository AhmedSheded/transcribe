#!/usr/bin/env bash
# Wrapper: activates the venv and runs the transcriber.
# Usage: ./transcribe.sh /path/to/video.mp4 [extra args...]
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Make the pip-installed CUDA libraries visible to CTranslate2.
PY_LIB="$("$DIR/venv/bin/python" -c 'import site,os;print(site.getsitepackages()[0])')"
export LD_LIBRARY_PATH="$PY_LIB/nvidia/cublas/lib:$PY_LIB/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"

exec "$DIR/venv/bin/python" "$DIR/transcribe.py" "$@"
