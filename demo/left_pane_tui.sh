#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# Left pane of the tui demo: tensixinet --live --tui on the real chips, beside tt-toplike's arcade view.
#
# tensixinet reads the chips with hwmon and tt-smi and opens no device, so it needs no gozer lease. --player none paces the picture at the speed
# of the music with no sound device, so the picture and the WAV that demo/build_tui.sh lays under the recording line up exactly. DEMO_SECONDS is the
# length of the soundtrack; the scene in demos.yaml runs a few seconds longer so the run can finish and write the WAV.
cd "$(dirname "$0")/.." || exit 1
clear
export COLORTERM=truecolor
python3 -m tensixinet --live --tui \
  --color always \
  --duration "${DEMO_SECONDS:-110}" \
  --player none \
  --out demo/assets/tui.wav \
  --log-dir demo/assets/logs_tui
# Hold the pane so the last frame stays on screen until the scene ends.
exec sleep 300
