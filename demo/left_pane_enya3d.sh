#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# Left pane of the enya3d demo: tensixinet --live --style enya beside tt-toplike's Die 3D view, with a model served on one board
# and sent traffic in the middle of the run.
#
# Settings (environment):
#   DEMO_SECONDS    length of the soundtrack (600 s is 10:00); the scene runs a few seconds longer so the run can finish and write the WAV
#   ENYA_KERNELS    empty for the CPU soundtrack, or a --kernels mode (chip) for the devices that have TT-Lang kernels; the run leases one board
#                   itself, so the model must be on the other board (see kernels/compat/env.sh for the toolchain it sets up)
#   TRAFFIC_LEAD    seconds before the first burst of prompts; TRAFFIC_BURST, TRAFFIC_REST, TRAFFIC_CYCLES and TRAFFIC_DECODE shape the bursts
#   ENDPOINT        the model's OpenAI-compatible endpoint; with nothing answering there, no traffic is sent
# tensixinet reads the chips with hwmon and tt-smi, so the CPU soundtrack needs no gozer lease. --player none paces the output at the speed
# of the music with no sound device, and --out writes the WAV that demo/build_enya3d.sh lays under the recording.
cd "$(dirname "$0")/.." || exit 1
clear
export COLORTERM=truecolor
ENDPOINT="${ENDPOINT:-http://127.0.0.1:8011}"
mkdir -p demo/assets
if curl -sf -m 3 "$ENDPOINT/v1/models" >/dev/null; then
  python3 demo/load_prompts.py --endpoint "$ENDPOINT" \
    --lead "${TRAFFIC_LEAD:-150}" --burst "${TRAFFIC_BURST:-50}" --rest "${TRAFFIC_REST:-20}" --cycles "${TRAFFIC_CYCLES:-5}" \
    --streams "${TRAFFIC_STREAMS:-12}" --decode-streams "${TRAFFIC_DECODE:-4}" >demo/assets/enya3d.traffic.log 2>&1 &
else
  echo "no model answering at $ENDPOINT: no traffic will be sent" >demo/assets/enya3d.traffic.log
fi
KERNELS=()
if [ -n "${ENYA_KERNELS:-}" ]; then
  KERNELS=(--kernels "$ENYA_KERNELS")
  source kernels/compat/env.sh || exit 1
fi
printf '\033[1;38;2;79;209;197m$\033[0m tensixinet --live --style enya %s--duration %s\n\n' "${ENYA_KERNELS:+--kernels $ENYA_KERNELS }" "${DEMO_SECONDS:-600}"
python3 -m tensixinet --live \
  --style enya \
  "${KERNELS[@]}" \
  --color always \
  --duration "${DEMO_SECONDS:-600}" \
  --player none \
  --out demo/assets/enya3d.wav \
  --log-dir demo/assets/logs_enya3d
# Hold the pane so the final log line stays on screen until the scene ends.
exec sleep 300
