#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# Build the "enya" demo: record tensixinet --live --style enya beside tt-toplike cycling through its views (--rotate), lay the soundtrack under it,
# and cut the video to 4:20 (DEMO_SECONDS, 260 by default).
#
#   demo/build_enya.sh     record + render + soundtrack + cut
#
set -euo pipefail
cd "$(dirname "$0")/.."
SECONDS_TOTAL="${DEMO_SECONDS:-260}"
export DEMO_SECONDS="$SECONDS_TOTAL"

tt-demo record enya
tt-demo compress demo/assets/enya.cast
tt-demo render enya --mp4
python3 -m tensixinet.media.mux demo/assets/enya.mp4 demo/assets/enya.wav demo/assets/enya.mux.mp4
ffmpeg -hide_banner -loglevel error -y -i demo/assets/enya.mux.mp4 -t "$SECONDS_TOTAL" -c copy demo/assets/enya.final.mp4
rm -f demo/assets/enya.mux.mp4
tt-demo verify enya
echo "soundtrack check:"
ffprobe -v error -show_entries stream=codec_type,codec_name,duration -of csv=p=0 demo/assets/enya.final.mp4
