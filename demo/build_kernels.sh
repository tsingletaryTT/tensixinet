#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# Build the "kernels" demo: record the split-pane scene (the soundtrack on four chips beside tt-toplike), then lay the soundtrack under it.
#
#   demo/build_kernels.sh     record + render + soundtrack
#
# The left pane takes a gozer lease on all four chips for the run, so nothing else may hold them. The recording ends before the
# chips' reset finishes; the worker that holds them exits when the pane closes and gozer releases them then.
set -euo pipefail
cd "$(dirname "$0")/.."

tt-demo record kernels
tt-demo compress demo/assets/kernels.cast
tt-demo render kernels --mp4
python3 -m tensixinet.media.mux demo/assets/kernels.mp4 demo/assets/kernels.wav demo/assets/kernels.final.mp4
tt-demo verify kernels
echo "soundtrack check:"
ffprobe -v error -show_entries stream=codec_type,codec_name,duration -of csv=p=0 demo/assets/kernels.final.mp4
