#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# Left pane of the "every feature" demo: tensixinet --live, reading the real chips every turn, with the kernel devices on the chips.
#
# `--kernels chips` takes one gozer lease on every chip the run plays, so the chips must be free; tt-toplike on the right only reads them. --player none paces the
# output at the speed of the music with no sound device, and --out still writes the WAV that
# demo/build.sh later muxes under the recording as its soundtrack. DEMO_SECONDS must match the
# scene duration in demos.yaml.
cd "$(dirname "$0")/.." || exit 1
clear
export COLORTERM=truecolor
echo '$ tensixinet --live --kernels chips --duration 110'
echo
echo "live: reading the real chips every turn (tt-smi -s) and playing as it goes, the kernel devices on the chips themselves."
echo "all eleven devices, patch-drawn cables, the coco bus, rests, turn-taking, EQ, loudness."
echo
python3 -m tensixinet --live \
  --kernels chips \
  --color always \
  --duration "${DEMO_SECONDS:-110}" \
  --player none \
  --out demo/assets/features.wav \
  --log-dir demo/assets/logs_features
# Hold the pane so the final log line stays on screen until the scene ends.
exec sleep 300
