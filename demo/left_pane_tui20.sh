#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# Left pane of the tui20 demo: tensixinet --live --tui in its default style (chiptune) on the real chips, read-only (no lease), CPU band.
# --player none paces the picture at the speed of the music with no sound device, so the picture and the WAV (--out, which the build script puts
# under the video) line up exactly; nothing plays on the speakers.
#   DEMO_SECONDS   length of the run (1200 s is 20:00); the scene runs a few seconds longer so the run can finish and write the WAV
cd "$(dirname "$0")/.." || exit 1
clear
export COLORTERM=truecolor
mkdir -p demo/assets
python3 -m tensixinet --live --tui \
  --color always \
  --duration "${DEMO_SECONDS:-1200}" \
  --player none \
  --out demo/assets/tui20.wav \
  --log-dir demo/assets/logs_tui20
# Hold the pane so the last frame stays on screen until the scene ends.
exec sleep 300
