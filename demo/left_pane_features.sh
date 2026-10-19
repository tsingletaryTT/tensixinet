#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# Left pane of the "every feature" demo: tensixinet --live, reading the real chips every turn.
#
# Read-only (tt-smi -s each turn), so this needs no gozer lease. --player none paces the
# output at the speed of the music with no sound device, and --out still writes the WAV that
# demo/build.sh later muxes under the recording as its soundtrack. DEMO_SECONDS must match the
# scene duration in demos.yaml.
cd "$(dirname "$0")/.." || exit 1
clear
echo '$ tensixinet --live --duration 110'
echo
echo "live: reading the real chips every turn (tt-smi -s, read-only) and playing as it goes."
echo "all eleven devices, patch-drawn cables between them, turn-taking, EQ, loudness."
echo
python3 -m tensixinet --live \
  --duration "${DEMO_SECONDS:-110}" \
  --player none \
  --out demo/assets/features.wav \
  --log-dir demo/assets/logs_features
# Hold the pane so the final log line stays on screen until the scene ends.
exec sleep 300
