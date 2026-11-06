#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# Left pane of the tui10 demo: tensixinet --live --tui in its default style, beside tt-toplike's Die 3D view, with a model served on one board
# and sent traffic in the middle of the run. The same events as the enya3d demo, with the terminal view in place of the log.
#
# Settings (environment):
#   DEMO_SECONDS    length of the soundtrack (600 s is 10:00); the scene runs a few seconds longer so the run can finish and write the WAV
#   TUI_KERNELS     empty for the CPU soundtrack, or a --kernels mode (chip) for the devices that have TT-Lang kernels; the run leases one board
#                   itself, so the model must be on the other board (see kernels/compat/env.sh for the toolchain it sets up)
#   TRAFFIC_LEAD    seconds before the first burst of prompts; TRAFFIC_BURST, TRAFFIC_REST, TRAFFIC_CYCLES and TRAFFIC_DECODE shape the bursts
#   ENDPOINT        the model's OpenAI-compatible endpoint; with nothing answering there, no traffic is sent
# --player none paces the picture at the speed of the music with no sound device, so the picture and the WAV line up exactly.
cd "$(dirname "$0")/.." || exit 1
clear
export COLORTERM=truecolor
ENDPOINT="${ENDPOINT:-http://127.0.0.1:8011}"
mkdir -p demo/assets
if curl -sf -m 3 "$ENDPOINT/v1/models" >/dev/null; then
  python3 demo/load_prompts.py --endpoint "$ENDPOINT" \
    --lead "${TRAFFIC_LEAD:-150}" --burst "${TRAFFIC_BURST:-50}" --rest "${TRAFFIC_REST:-20}" --cycles "${TRAFFIC_CYCLES:-5}" \
    --streams "${TRAFFIC_STREAMS:-12}" --decode-streams "${TRAFFIC_DECODE:-4}" >demo/assets/tui10.traffic.log 2>&1 &
else
  echo "no model answering at $ENDPOINT: no traffic will be sent" >demo/assets/tui10.traffic.log
fi
KERNELS=()
if [ -n "${TUI_KERNELS:-}" ]; then
  KERNELS=(--kernels "$TUI_KERNELS")
  source kernels/compat/env.sh || exit 1
fi
python3 -m tensixinet --live --tui \
  "${KERNELS[@]}" \
  --color always \
  --duration "${DEMO_SECONDS:-600}" \
  --player none \
  --out demo/assets/tui10.wav \
  --log-dir demo/assets/logs_tui10
# Hold the pane so the last frame stays on screen until the scene ends.
exec sleep 300
