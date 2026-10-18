#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# Build the tensixinet demo: record the split-pane scene, then lay the chorus under it.
#
#   demo/build.sh            record + render + soundtrack
#
# Reads telemetry only (tt-toplike --backend hybrid, tt-smi -s) and sends one prompt
# to the LLM endpoint, so no gozer lease is needed. Needs the endpoint from
# demo/left_pane.sh (default http://127.0.0.1:8011) to be up; if it is not, the
# chorus still plays and the log says the LLM was unavailable.
set -euo pipefail
cd "$(dirname "$0")/.."

tt-demo record chorus
tt-demo compress demo/assets/chorus.cast
tt-demo render chorus --mp4
python3 -m tensixinet.media.mux demo/assets/chorus.mp4 demo/assets/chorus.wav demo/assets/chorus.final.mp4
tt-demo verify chorus
echo "soundtrack check:"
ffprobe -v error -show_entries stream=codec_type,codec_name,duration -of csv=p=0 demo/assets/chorus.final.mp4
