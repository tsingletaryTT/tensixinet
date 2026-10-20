#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# Build the "stereo" demo: record the split-pane scene (the stereo band and evictions on the real chips, with a served model prompted in bursts, beside tt-toplike), then lay the soundtrack under it.
# A model must already be serving on port 8011 (see left_pane_stereo.sh).
#
#   demo/build_stereo.sh     record + render + soundtrack
#
set -euo pipefail
cd "$(dirname "$0")/.."

tt-demo record stereo
tt-demo compress demo/assets/stereo.cast
tt-demo render stereo --mp4
python3 -m tensixinet.media.mux demo/assets/stereo.mp4 demo/assets/stereo.wav demo/assets/stereo.final.mp4
tt-demo verify stereo
echo "soundtrack check:"
ffprobe -v error -show_entries stream=codec_type,codec_name,duration -of csv=p=0 demo/assets/stereo.final.mp4
