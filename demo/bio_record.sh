#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# Record tt-bio-demo full screen (the 2x2 quad, all four chips folding) with tensixinet's music of the same run laid under it.
#
#   BIO_PATCH=logs/patch_20261021_101821.json demo/bio_record.sh          four minutes
#   BIO_SECONDS=120 BIO_PATCH=... demo/bio_record.sh                      a shorter one
#
# The video is tt-bio-demo's own recording script as demo/bio_capture.sh (OBS first, then the booth fullscreen over it, and a check that the screen is clean),
# run under one gozer lease on all four chips (its scripts/hw-run.sh). No --questions, so every chip folds. OBS needs its screen-capture portal
# granted once per login session: if a share dialog appears, choose the screen. `tensixinet --live` starts as soon as the recording is clean, reads
# the same chips with tt-smi, and replays BIO_PATCH. The music is muxed under the video, delayed by how long tensixinet takes to make its first
# segment (read from its log), since the video starts a couple of seconds after the clean check and the music a few seconds before its first sound.
# The kernel devices stay on the CPU: the booth holds the chips. Output: build/bio_rec/bio.master.mp4 and bio.mp4.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
BIO="${TT_BIO_DEMO_DIR:-$HOME/code/tt-bio-demo}"
SECONDS_TOTAL="${BIO_SECONDS:-240}"
PATCH="${BIO_PATCH:?set BIO_PATCH to a saved patch (logs/patch_*.json)}"
OUT="$PWD/build/bio_rec"
rm -rf "$OUT"; mkdir -p "$OUT/video"
[ -f "$PATCH" ] || { echo "no patch at $PATCH"; exit 1; }
[ -x "$BIO/scripts/hw-run.sh" ] || { echo "no tt-bio-demo checkout at $BIO (set TT_BIO_DEMO_DIR)"; exit 1; }

( cd "$BIO" && OUT_DIR="$OUT/video" scripts/hw-run.sh --chips 4 --reason "tensixinet: record tt-bio-demo with the band" -- \
    "$ROOT/demo/bio_capture.sh" "$SECONDS_TOTAL" ) > "$OUT/record.log" 2>&1 &
REC=$!
trap 'kill -INT $REC 2>/dev/null' EXIT

echo "waiting for the recording to be clean (OBS, the chips, the booth)"
until grep -q "capturing" "$OUT/record.log" 2>/dev/null; do
  kill -0 "$REC" 2>/dev/null || { echo "the recording ended early; see $OUT/record.log"; tail -8 "$OUT/record.log"; exit 1; }
  sleep 0.3
done
T_CLEAN=$(date +%s.%N)
echo "clean; starting the music"
python3 -m tensixinet --live --patch "$PATCH" --duration $((SECONDS_TOTAL + 8)) --no-banner --color never \
    --out "$OUT/music.wav" --log-dir "$OUT/logs_music" > "$OUT/music.out" 2>&1 &
MUSIC=$!
wait "$REC"; REC_STATUS=$?
kill -INT "$MUSIC" 2>/dev/null; wait "$MUSIC" 2>/dev/null
trap - EXIT
[ -f "$OUT/video/quad-master.mp4" ] || { echo "no video (status $REC_STATUS); see $OUT/record.log"; tail -8 "$OUT/record.log"; exit 1; }

# the music's first sound is at its log's "segment 0" time; the video's first frame is 2 s after the clean check
FIRST=$(grep -m1 -oE '\[[0-9]+:[0-9.]+\] segment 0:' "$OUT"/logs_music/*.txt | head -1 | sed -E 's/\[([0-9]+):([0-9.]+)\].*/\1 \2/' | awk '{print $1*60+$2}')
FIRST=${FIRST:-5.5}
DELAY=$(python3 -c "print(round($FIRST - 2.0, 2))")
echo "music's first sound at $FIRST s into its run; delaying it $DELAY s"
cp "$OUT/video/quad-master.mp4" "$OUT/bio.master.mp4"
if python3 -c "import sys; sys.exit(0 if $DELAY >= 0 else 1)"; then AIN=(-itsoffset "$DELAY" -i "$OUT/music.wav"); else AIN=(-ss "${DELAY#-}" -i "$OUT/music.wav"); fi
ffmpeg -hide_banner -loglevel error -y -i "$OUT/video/quad-master.mp4" "${AIN[@]}" -map 0:v -map 1:a \
    -c:v libx264 -crf 24 -preset medium -pix_fmt yuv420p -r 30 -movflags +faststart -c:a aac -b:a 192k -shortest "$OUT/bio.mp4"
ffprobe -v error -show_entries stream=codec_type,codec_name,duration -of csv=p=0 "$OUT/bio.mp4"
ls -la "$OUT"/bio.master.mp4 "$OUT"/bio.mp4 | awk '{print $5, $9}'
