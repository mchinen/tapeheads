#!/usr/bin/env bash
# Run formatting, lint, type, and unit checks from the project environment.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

.venv/bin/pyink --check tapeheads tools tests
.venv/bin/ruff check tapeheads tools tests
.venv/bin/pytype
.venv/bin/python -m unittest discover -s tests -q
