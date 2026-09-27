#!/usr/bin/env bash
# Install MuQ's optional dependencies with matching CPU-only Torch packages.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
bash tools/setup_analysis.sh
.venv/bin/python -m pip install --cache-dir .cache/pip \
  torchaudio==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install --cache-dir .cache/pip muq==0.1.0
