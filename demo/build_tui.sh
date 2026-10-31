#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# Build the "tui" demo: record tensixinet --live --tui beside tt-toplike (arcade view), lay the soundtrack under it and cut the video to DEMO_SECONDS.
#
#   demo/build_tui.sh     record + render + soundtrack + cut
#
set -euo pipefail
cd "$(dirname "$0")/.."
SECONDS_TOTAL="${DEMO_SECONDS:-110}"
export DEMO_SECONDS="$SECONDS_TOTAL"

tt-demo record tui
tt-demo compress demo/assets/tui.cast
tt-demo render tui --mp4
python3 -m tensixinet.media.mux demo/assets/tui.mp4 demo/assets/tui.wav demo/assets/tui.mux.mp4
ffmpeg -hide_banner -loglevel error -y -i demo/assets/tui.mux.mp4 -t "$SECONDS_TOTAL" -c copy demo/assets/tui.final.mp4
rm -f demo/assets/tui.mux.mp4
tt-demo verify tui
ffprobe -v error -show_entries stream=codec_type,codec_name,duration -of csv=p=0 demo/assets/tui.final.mp4
