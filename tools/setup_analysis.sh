#!/usr/bin/env bash
# Create the CPU-only environment used by the first-layer prototype.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
python3 -m venv .venv
.venv/bin/python -m pip install --cache-dir .cache/pip --upgrade pip
.venv/bin/python -m pip install --cache-dir .cache/pip \
  torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install --cache-dir .cache/pip -r requirements.txt
