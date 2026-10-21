#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# Watch tt-bio-demo fold proteins while tensixinet plays the chips doing the work.
#
#   demo/bio_session.sh                  about four minutes
#   BIO_SECONDS=120 demo/bio_session.sh  a shorter one
#
# Two windows open. The tt-bio-demo booth (a normal GTK window, `--windowed`) folds on three chips while the fourth is reserved to answer
# affinity questions (`--questions`), under one gozer lease on all four chips (tt-bio-demo's own scripts/hw-run.sh). A terminal beside it
# runs `tensixinet --live`, which reads the same chips with tt-smi (read-only, no lease) and plays them through the speakers. Tile the two
# windows side by side (Meta+Left and Meta+Right in KDE). The kernel devices stay on the CPU, because the booth holds the chips.
#
# TT_BIO_DEMO_DIR is the tt-bio-demo checkout (default ~/code/tt-bio-demo). Nothing here touches its installed package or its service.
set -uo pipefail
cd "$(dirname "$0")/.."
BIO="${TT_BIO_DEMO_DIR:-$HOME/code/tt-bio-demo}"
SECONDS_TOTAL="${BIO_SECONDS:-240}"
SOCKET="${XDG_RUNTIME_DIR:-/tmp}/tt-bio-demo/runner.sock"
mkdir -p build

[ -x "$BIO/scripts/hw-run.sh" ] || { echo "no tt-bio-demo checkout at $BIO (set TT_BIO_DEMO_DIR)"; exit 1; }

# 1. the booth, under a lease on every chip; it ends itself a little after the music does
( cd "$BIO" && timeout -s INT $((SECONDS_TOTAL + 120)) scripts/hw-run.sh --chips 4 \
    --reason "tensixinet listening session: tt-bio-demo with the band" -- scripts/run-demo.sh --questions --windowed ) > build/bio_booth.log 2>&1 &
BOOTH=$!
trap 'kill -INT $BOOTH 2>/dev/null' EXIT

# 2. wait for the booth to come up (the lease, then the daemon's socket)
echo "waiting for the booth (the chips, then the model, take a minute or two)"
for _ in $(seq 1 240); do
  [ -S "$SOCKET" ] && break
  kill -0 "$BOOTH" 2>/dev/null || { echo "the booth ended early; see build/bio_booth.log"; tail -5 build/bio_booth.log; exit 1; }
  sleep 1
done
[ -S "$SOCKET" ] || { echo "the booth's socket never appeared; see build/bio_booth.log"; exit 1; }
sleep 20   # the first folds start

# 3. the music, in a terminal beside it
echo "starting the music for $SECONDS_TOTAL s"
konsole --separate -e bash -c "cd '$PWD' && ./tensixinet.py --live --duration $SECONDS_TOTAL; echo; echo 'done; this window closes in 30 s'; sleep 30" &
sleep "$SECONDS_TOTAL"
echo "stopping the booth"
kill -INT "$BOOTH" 2>/dev/null
wait "$BOOTH" 2>/dev/null
echo "done"
