#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# Render the site's listening clips (site/assets/audio/*.mp3) from the real chips, with the devices that have TT-Lang kernels running on the
# chips themselves (`--kernels chips`: one gozer lease on every chip the run plays, taken and released by the run). Every clip is stereo, from
# patch seed 5, so the clips can be compared. Each run takes about a minute: the lease, the render, and the 40 s the chips take to reset.
#
#   demo/make_clips.sh              all the clips
#   demo/make_clips.sh enya blues   only the named ones
#
# Reads the chips with tt-smi like every other run, so run it when nobody else needs the boards.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=site/assets/audio
WORK=${CLIP_WORK:-build/clips}
mkdir -p "$OUT" "$WORK"

clip() {   # clip NAME SECONDS ARGS...
  local name=$1 secs=$2; shift 2
  if [ -n "${WANT:-}" ] && ! grep -qw -- "$name" <<<"$WANT"; then return; fi
  echo "== $name"
  python3 -m tensixinet --patch-seed 5 --duration "$secs" --kernels chips --no-banner --color never \
      --out "$WORK/$name.wav" --log-dir "$WORK/logs_$name" "$@" > "$WORK/$name.out" 2>&1 \
    || { echo "FAILED: $name (see $WORK/$name.out)"; return; }
  ffmpeg -v error -y -i "$WORK/$name.wav" -codec:a libmp3lame -b:a 128k "$OUT/$name.mp3" && echo "wrote $OUT/$name.mp3"
}

WANT="$*"
clip all 30
clip all_nocables 30 --no-cables
clip chiptune 40 --style chiptune
clip enya 40 --style enya
clip blues 40 --style blues
clip aphex 40 --style aphex
clip choir 30 --only ttchoir
clip avdog 30 --only ttavdog
clip gong 30 --only gong
clip trax 30 --only ttrax
