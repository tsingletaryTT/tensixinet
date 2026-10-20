#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# Left pane of the stereo and evict demo: tensixinet --live on four synthetic chips that cycle through load.
#
# Synthetic chips, because the real ones sit idle and an idle chip never evicts. --mock-load 60 takes each chip through rest,
# load and rest every 60 s, offset chip from chip, so the log shows evictions and fills. It needs no chips and no gozer lease.
# --color always keeps the jack colours on for the recorder. DEMO_SECONDS must be shorter than the scene.
cd "$(dirname "$0")/.." || exit 1
clear
export COLORTERM=truecolor
printf '\033[1;38;2;79;209;197m$\033[0m tensixinet --live --mock-chips 4 --mock-load 60 --duration %s\n\n' "${DEMO_SECONDS:-100}"
python3 -m tensixinet --live \
  --mock-chips 4 --mock-load 60 \
  --patch-seed 4 \
  --color always \
  --duration "${DEMO_SECONDS:-100}" \
  --player none \
  --out demo/assets/stereo.wav \
  --log-dir demo/assets/logs_stereo
# Hold the pane so the final log line stays on screen until the scene ends.
exec sleep 300
