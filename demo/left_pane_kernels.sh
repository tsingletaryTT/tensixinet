#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# Left pane of the "kernels" demo: tensixinet --live with the TT-Lang kernels running on all four chips.
#
# `--kernels chips` takes one gozer lease on every chip the run plays (both boards here) and runs each chip's gong, shimmer, drone
# ripple, AVDog and ttcoco mix, and the final mixer, on that chip's own kernels; the other devices stay on the CPU. tt-toplike on the
# right reads the same chips read-only (hybrid backend). --player none paces the output at the speed of the music with no sound
# device, and --out writes the WAV that demo/build_kernels.sh muxes under the recording. --color always keeps the Ciat-Lonbarde
# jack colours on even though the recorder, not a terminal, is what is watching. DEMO_SECONDS must be shorter than the scene.
cd "$(dirname "$0")/.." || exit 1
clear
export COLORTERM=truecolor
printf '\033[1;38;2;79;209;197m$\033[0m tensixinet --live --kernels chips --duration %s\n\n' "${DEMO_SECONDS:-100}"
python3 -m tensixinet --live \
  --kernels chips \
  --color always \
  --duration "${DEMO_SECONDS:-100}" \
  --player none \
  --out demo/assets/kernels.wav \
  --log-dir demo/assets/logs_kernels
# Hold the pane so the final log line stays on screen until the scene ends.
exec sleep 300
