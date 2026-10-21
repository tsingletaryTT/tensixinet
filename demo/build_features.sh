#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# Build the "features" demo: record the split-pane scene (tensixinet --live with the kernel devices on all four chips, beside tt-toplike's arcade view), then lay the soundtrack under it.
# The run takes a gozer lease on every chip, so the chips must be free (see left_pane_features.sh).
#
#   demo/build_features.sh     record + render + soundtrack
#
set -euo pipefail
cd "$(dirname "$0")/.."

tt-demo record features
tt-demo compress demo/assets/features.cast
tt-demo render features --mp4
python3 -m tensixinet.media.mux demo/assets/features.mp4 demo/assets/features.wav demo/assets/features.final.mp4
tt-demo verify features
echo "soundtrack check:"
ffprobe -v error -show_entries stream=codec_type,codec_name,duration -of csv=p=0 demo/assets/features.final.mp4
