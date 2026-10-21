#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# Record what a chip reset sounds like: tensixinet --live reads all four chips while a gozer lease on them is released. `gozer release` resets
# exactly the chips the lease held (the same reset `tt-smi -r` makes, taken through the lease so nobody else's chips are touched), and the chips
# stay out of tt-smi's reach for about 40 s. The band runs on the CPU here: the kernels need the chips, and the chips are the ones being reset.
#
#   demo/make_reset_clip.sh            site/assets/audio/reset.mp3 and build/clips/logs_reset
#
# The chips must be free (gozer status). Reads with tt-smi like every run; the lease is the only thing that touches the chips.
set -uo pipefail
cd "$(dirname "$0")/.."
WORK=${CLIP_WORK:-build/clips}
OUT=site/assets/audio/reset.mp3
SECONDS_TOTAL=${RESET_SECONDS:-110}
BEFORE=${RESET_BEFORE_S:-30}
mkdir -p "$WORK"

setsid sleep $((SECONDS_TOTAL + 600)) >/dev/null 2>&1 &
HOLDER=$!
trap 'kill $HOLDER 2>/dev/null' EXIT
LEASE=$(gozer acquire --json --chips 4 --who "claude:tensixinet" --reason "record what a chip reset sounds like" --expect 10m --owner-pid "$HOLDER" \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["lease_id"])') || { echo "could not lease the chips"; exit 1; }
echo "lease $LEASE; reading the chips for $SECONDS_TOTAL s, releasing it after $BEFORE s"

python3 -m tensixinet --live --patch-seed 5 --duration "$SECONDS_TOTAL" --player none --color never --no-banner \
    --out "$WORK/reset.wav" --log-dir "$WORK/logs_reset" > "$WORK/reset.out" 2>&1 &
LIVE=$!
sleep "$BEFORE"
echo "releasing $LEASE (this resets the chips)"
gozer release "$LEASE" > "$WORK/reset_release.out" 2>&1
echo "released at $(date +%T)"
wait "$LIVE"
ffmpeg -v error -y -i "$WORK/reset.wav" -codec:a libmp3lame -b:a 128k "$OUT" && echo "wrote $OUT"
