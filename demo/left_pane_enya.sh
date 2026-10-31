#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# Left pane of the enya demo: tensixinet --live --style enya on the real chips, beside tt-toplike cycling through its views.
#
# tensixinet reads the chips with hwmon and tt-smi and opens no device, so it needs no gozer lease. --player none paces the output at the speed of the music with
# no sound device, and --out writes the WAV that demo/build_enya.sh lays under the recording. DEMO_SECONDS is the length of the soundtrack (260 s is 4:20); the scene in
# demos.yaml runs a few seconds longer so the run can finish and write the WAV, and the build script cuts the video to DEMO_SECONDS.
cd "$(dirname "$0")/.." || exit 1
clear
export COLORTERM=truecolor
printf '\033[1;38;2;79;209;197m$\033[0m tensixinet --live --style enya --duration %s\n\n' "${DEMO_SECONDS:-260}"
python3 -m tensixinet --live \
  --style enya \
  --color always \
  --duration "${DEMO_SECONDS:-260}" \
  --player none \
  --out demo/assets/enya.wav \
  --log-dir demo/assets/logs_enya
# Hold the pane so the final log line stays on screen until the scene ends.
exec sleep 300
