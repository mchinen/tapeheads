#!/usr/bin/env bash
# Install the pinned yt-dlp executable into this project's .tools directory.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
yt_dlp_version="2026.08.19"
tool_directory="${project_root}/.tools"
mkdir -p "${tool_directory}"
curl --fail --location --retry 2 \
  "https://github.com/yt-dlp/yt-dlp/releases/download/${yt_dlp_version}/yt-dlp" \
  --output "${tool_directory}/yt-dlp.partial"
chmod +x "${tool_directory}/yt-dlp.partial"
"${tool_directory}/yt-dlp.partial" --version
mv "${tool_directory}/yt-dlp.partial" "${tool_directory}/yt-dlp"
