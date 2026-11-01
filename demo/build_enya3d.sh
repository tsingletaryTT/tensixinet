#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# Build the "enya3d" demo: tensixinet --live --style enya beside tt-toplike's Die 3D view, a model sent traffic in the middle, the run's own
# audio under it, cut to DEMO_SECONDS (600 by default). The scene's `duration` in demos.yaml must be longer than DEMO_SECONDS.
#
#   demo/build_enya3d.sh                         10:00 with the TT-Lang kernels on one leased board (ENYA_KERNELS=chip); the model is served
#                                                on the other board first, and gets bursts of prompts from 2:30 to 8:20
#   ENYA_KERNELS= demo/build_enya3d.sh           the CPU soundtrack (shorten DEMO_SECONDS, and the scene's duration, to make a shorter one)
#
# The picture is rendered at a larger font than the default (MP4_FONT_SIZE) and encoded at CRF 16; the audio is the run's WAV as 320 kbit/s AAC.
set -euo pipefail
cd "$(dirname "$0")/.."
export DEMO_SECONDS="${DEMO_SECONDS:-600}"
export ENYA_KERNELS="${ENYA_KERNELS-chip}"
export MP4_FONT_SIZE="${MP4_FONT_SIZE:-16}"

tt-demo record enya3d
tt-demo compress demo/assets/enya3d.cast
tt-demo render enya3d --mp4
ffmpeg -hide_banner -loglevel error -y -i demo/assets/enya3d.mp4 -i demo/assets/enya3d.wav -map 0:v -map 1:a \
  -t "$DEMO_SECONDS" -c:v libx264 -preset slow -crf 16 -pix_fmt yuv420p -c:a aac -b:a 320k -shortest demo/assets/enya3d.final.mp4
tt-demo verify enya3d
echo "check:"
ffprobe -v error -show_entries stream=codec_type,codec_name,width,height,duration -of csv=p=0 demo/assets/enya3d.final.mp4
