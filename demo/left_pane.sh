#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# Left pane of the tensixinet demo: prompt the LLM, then play the chips and say why.
#
# The prompt goes to the LLM first so its load is on the chips when telemetry is
# sampled for the chorus. The WAV is later muxed under the recording as its soundtrack
# (demo/build.sh). DEMO_SECONDS must match the scene duration in demos.yaml.
cd "$(dirname "$0")/.." || exit 1
clear
export COLORTERM=truecolor
PROMPT="In a few sentences, describe what four hardware synthesizers sound like when each one is played by a different chip."
printf '$ python3 -m tensixinet "%s"\n\n' "$PROMPT"
# demo/chorus.patch.json was drawn from this machine's chips (seed picked for one-bar turns,
# a low floor and a lively motif), so rebuilding the demo plays the same music. Delete the
# --patch line to draw a fresh one-of-a-kind patch every time.
python3 -m tensixinet "$PROMPT" \
  --color always \
  --patch demo/chorus.patch.json \
  --duration "${DEMO_SECONDS:-50}" \
  --out demo/assets/chorus.wav \
  --log-dir demo/assets/logs
# Hold the pane so the log stays on screen until the scene ends.
exec sleep 300
