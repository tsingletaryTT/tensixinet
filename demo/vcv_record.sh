#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# Record tensixinet and a VCV Rack voice playing on top of it, in one 1080p frame with the sound of both.
#
#   demo/vcv_record.sh                       two minutes
#   VCV_SECONDS=45 demo/vcv_record.sh        a shorter one
#   VCV_LISTEN=0 demo/vcv_record.sh          do not also play it through your speakers
#
# Top left: `tensixinet --live --midi-port virtual` reading the real chips (tt-smi only, no lease). Top right: `aseqdump` showing the MIDI it sends. Bottom: VCV Rack Free with
# vcv/tensixinet-voice.vcv, a monophonic voice with a dark, resonant low-pass filter, and its scope.
#
# Nothing here touches your desktop or your Rack settings. Everything runs on a private X server (Xvfb, software GL), so no screen-share dialog and no lock screen to get in the
# way, and the picture is grabbed from that server. Rack gets its own user folder (build/vcv_rec/rack-user, with your plugins linked in), so its settings, log and autosave stay
# out of ~/.local/share/Rack2. The sound goes to a private null sink, `tsx_demo`, that both programs play into, and ffmpeg records its monitor, so the mix is what the
# programs really played together. VCV_LISTEN=1 (the default) also loops that sink to your default output so you can hear the take.
#
# Needs: Xvfb, xterm, xdotool, ffmpeg, pactl (PulseAudio or PipeWire-pulse), aseqdump, tar and zstd, VCV Rack Free (VCV_RACK_DIR, default ~/code/vcv-cv-harness/Rack2Free) with its
# plugins in ~/.local/share/Rack2/plugins-lin-x64, and Python with the midi extra (VCV_PYTHON, default python3). Output: build/vcv_rec/vcv.mp4.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
SECONDS_TOTAL="${VCV_SECONDS:-120}"
RACK_DIR="${VCV_RACK_DIR:-$HOME/code/vcv-cv-harness/Rack2Free}"
PY="${VCV_PYTHON:-python3}"
LISTEN="${VCV_LISTEN:-1}"
SEED="${VCV_PATCH_SEED:-5}"
OFFSET_MS="${VCV_MIDI_OFFSET_MS:-0}"
OUT="$ROOT/build/vcv_rec"
SINK=tsx_demo
rm -rf "$OUT"; mkdir -p "$OUT/rack-user"
[ -x "$RACK_DIR/Rack" ] || { echo "no Rack at $RACK_DIR/Rack (set VCV_RACK_DIR)"; exit 1; }
[ -d "$HOME/.local/share/Rack2/plugins-lin-x64" ] || { echo "no Rack plugins in ~/.local/share/Rack2/plugins-lin-x64"; exit 1; }
for t in Xvfb xterm xdotool ffmpeg pactl aseqdump tar zstd; do command -v "$t" >/dev/null || { echo "missing $t"; exit 1; }; done
ln -s "$HOME/.local/share/Rack2/plugins-lin-x64" "$OUT/rack-user/plugins-lin-x64"
# a new user folder would open Rack's welcome tips and check for updates
echo '{"showTipsOnLaunch": false, "checkVersion": false, "skipLoginOnLaunch": true, "uiTheme": "dark"}' > "$OUT/rack-user/settings.json"

PIDS=(); MODS=()
cleanup() {
  trap - EXIT
  for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null; done
  for m in "${MODS[@]:-}"; do [ -n "$m" ] && pactl unload-module "$m" 2>/dev/null; done
}
trap cleanup EXIT

# 1. a private X server, on a free display number
( Xvfb -ac -screen 0 1920x1080x24 -displayfd 9 9> "$OUT/display" > "$OUT/xvfb.log" 2>&1 ) &
PIDS+=($!)
for _ in $(seq 1 50); do [ -s "$OUT/display" ] && break; sleep 0.1; done
[ -s "$OUT/display" ] || { echo "Xvfb did not start; see $OUT/xvfb.log"; exit 1; }
export DISPLAY=":$(cat "$OUT/display")"
unset XAUTHORITY WAYLAND_DISPLAY
echo "private display $DISPLAY"

# 2. the private sound sink (and, to hear the take, a loop to your speakers)
MODS+=($(pactl load-module module-null-sink sink_name=$SINK sink_properties=device.description=$SINK))
if [ "$LISTEN" = 1 ]; then MODS+=($(pactl load-module module-loopback source=$SINK.monitor latency_msec=40)); fi

# 3. tensixinet (top left) and the MIDI it sends (top right)
xterm -T tensixinet -geometry 160x32+0+0 -b 0 -bg '#092221' -fg '#F1F8F8' -fa 'DejaVu Sans Mono' -fs 10 -e bash -c \
  "cd '$ROOT' && PULSE_SINK=$SINK $PY -m tensixinet --live --duration $((SECONDS_TOTAL + 40)) --patch-seed $SEED --midi-port virtual --midi-offset-ms $OFFSET_MS --color always --no-banner --out '$OUT/tensixinet.wav' --log-dir '$OUT/logs'; sleep 3" &
PIDS+=($!)
echo "waiting for tensixinet's MIDI port"
DEVICE=""
for _ in $(seq 1 120); do
  DEVICE=$($PY -c "from tensixinet.score.vcv import alsa_input_name; print(alsa_input_name() or '')" 2>/dev/null)
  [ -n "$DEVICE" ] && break; sleep 0.5
done
[ -n "$DEVICE" ] || { echo "tensixinet never opened its MIDI port (is the midi extra installed? pip install -e .[midi])"; exit 1; }
echo "port: $DEVICE"
PORT=${DEVICE##* }   # "128:0"
xterm -T "MIDI from tensixinet" -geometry 80x32+1280+0 -b 0 -bg '#092221' -fg '#74C5DF' -fa 'DejaVu Sans Mono' -fs 10 -e bash -c "aseqdump -p $PORT" &
PIDS+=($!)

# 4. Rack, with a copy of the patch that names this port and this sink
$PY -m tensixinet.score.vcv "$OUT/voice.vcv" --device "$DEVICE" --audio-device $SINK || exit 1
( cd "$RACK_DIR" && tail -f /dev/null | LD_LIBRARY_PATH=. exec ./Rack -u "$OUT/rack-user" "$OUT/voice.vcv" > "$OUT/rack.log" 2>&1 ) &
PIDS+=($!)
for _ in $(seq 1 100); do
  WIN=$(xdotool search --name 'VCV Rack' 2>/dev/null | head -1); [ -n "$WIN" ] && break; sleep 0.3
done
[ -n "$WIN" ] || { echo "Rack never opened a window; see $OUT/rack.log"; exit 1; }
sleep 3
xdotool windowmove "$WIN" 0 540; xdotool windowsize "$WIN" 1920 540
sleep 3

# 5. record: the picture of the private display and the sink's monitor, together
echo "recording $SECONDS_TOTAL s"
ffmpeg -hide_banner -loglevel error -y -f x11grab -framerate 30 -draw_mouse 0 -video_size 1920x1080 -i "$DISPLAY" \
  -f pulse -i $SINK.monitor -t "$SECONDS_TOTAL" \
  -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p -r 30 -c:a aac -b:a 192k -movflags +faststart "$OUT/vcv.mp4"
echo "recorded"
ffprobe -v error -show_entries stream=codec_type,codec_name,width,height,duration -of csv=p=0 "$OUT/vcv.mp4"
ls -la "$OUT/vcv.mp4" | awk '{print $5, $9}'
