#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# Build the folder to push to a Hugging Face Space: the code the Space runs and nothing else (no video, audio, screenshots, tests or docs).
#
#   space/assemble.sh [DEST]      DEST defaults to build/space
#
# DEST is replaced if it is empty, absent or was made by this script before (it holds a .tensixinet-space marker). Anything else is left
# alone and the script stops. It pushes nothing: copy or `git push` DEST to the Space yourself.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
dest="${1:-$here/build/space}"
marker=".tensixinet-space"

if [ -e "$dest" ] && [ -n "$(ls -A "$dest")" ] && [ ! -e "$dest/$marker" ]; then
  echo "assemble: $dest exists, is not empty and was not made by this script; not touching it" >&2
  exit 2
fi
rm -rf "$dest"
mkdir -p "$dest/app" "$dest/demo" "$dest/kernels"

cp -R "$here/tensixinet" "$dest/tensixinet"
cp "$here/app/gradio_app.py" "$dest/app/"
cp "$here/demo/chorus.patch.json" "$dest/demo/"
cp "$here"/kernels/_common.py "$here"/kernels/pool.py "$here"/kernels/worker.py "$here"/kernels/tt*.py "$dest/kernels/"
cp "$here/pyproject.toml" "$here/LICENSE" "$dest/"
cp "$here/space/Dockerfile" "$here/space/README.md" "$dest/"
find "$dest" -name __pycache__ -type d -prune -exec rm -rf {} +
touch "$dest/$marker"
echo "assembled $dest ($(du -sk "$dest" | cut -f1) KB)"
