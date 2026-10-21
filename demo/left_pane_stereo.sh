#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# Left pane of the stereo and evict demo: tensixinet --live on the real chips while a model on one board is loaded and then left alone.
#
# A model (Qwen3-Coder-Next, started by hand on one board, port 8011) sits on two of the chips. demo/load_prompts.py sends it bursts of
# prompts, so those chips load and then fall back to rest, and the eviction fires on real readings. The other two chips stay idle and
# never evict. tensixinet reads the chips with tt-smi and opens no device, so it needs no lease of its own. --color always keeps
# the jack colours on for the recorder. DEMO_SECONDS must be shorter than the scene.
cd "$(dirname "$0")/.." || exit 1
clear
export COLORTERM=truecolor
# three bursts: the model's draw is spiky (120 to 170 W for a moment, 45 to 60 W between), and the loop reads a chip about once a turn,
# so a recording needs several chances for a read to land in a load and again in the rest after it
python3 demo/load_prompts.py --lead 6 --burst 24 --rest 20 --cycles 3 > /dev/null 2>&1 &
LOAD=$!
trap 'kill $LOAD 2>/dev/null' EXIT
printf '\033[1;38;2;79;209;197m$\033[0m tensixinet --live --duration %s   \033[2m# a model on two chips, prompted in bursts\033[0m\n\n' "${DEMO_SECONDS:-120}"
python3 -m tensixinet --live \
  --color always \
  --duration "${DEMO_SECONDS:-120}" \
  --player none \
  --out demo/assets/stereo.wav \
  --log-dir demo/assets/logs_stereo
# Hold the pane so the final log line stays on screen until the scene ends.
exec sleep 300
