#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# Build the "tui20" demo: 20 minutes of tensixinet --live --tui (a quarter of the width) beside tt-toplike-busted --rotate (three quarters), tt-toplike-busted polling every
# 50 ms, and the run's own WAV (no sound device, so it lines up exactly) muxed under the picture. The scene's `duration` must exceed DEMO_SECONDS.
# Picture at MP4_FONT_SIZE, CRF 16 (high quality), audio 320 kbit/s AAC.
set -euo pipefail
cd "$(dirname "$0")/.."
export DEMO_SECONDS="${DEMO_SECONDS:-1200}"
export MP4_FONT_SIZE="${MP4_FONT_SIZE:-16}"

tt-demo record tui20
tt-demo compress demo/assets/tui20.cast
tt-demo render tui20 --mp4
ffmpeg -hide_banner -loglevel error -y -i demo/assets/tui20.mp4 -i demo/assets/tui20.wav -map 0:v -map 1:a \
  -t "$DEMO_SECONDS" -c:v libx264 -preset slow -crf 16 -pix_fmt yuv420p -c:a aac -b:a 320k -shortest demo/assets/tui20.final.mp4
tt-demo verify tui20
echo "check:"
ffprobe -v error -show_entries stream=codec_type,codec_name,width,height,duration -of csv=p=0 demo/assets/tui20.final.mp4
