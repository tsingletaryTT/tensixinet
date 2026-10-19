#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Turn the audio a chip computed (`cycle.py --save`) into a video: for each kernel, the chip's waveform, the numpy device's
waveform and the chip's spectrogram, with the chip's audio as the soundtrack and the measured difference in the caption.

    python3 kernels/make_video.py --runs RUNS/board1 --chip 0 --label "chip 2, board 1" --out hardware.mp4

It reads files only (no device, no `ttnn`) and needs `ffmpeg`.
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

KERNELS = ["ttshock", "ttshimmer", "ttgong", "ttavdog", "ttmix", "ttcoco"]
BLURB = {
    "ttshock": "the drone's shock wave",
    "ttshimmer": "the ultrasonic shimmer",
    "ttgong": "a ringing gong strike",
    "ttavdog": "the AVDog's two voices",
    "ttmix": "the chips' mixer",
    "ttcoco": "eight parallel delay taps, summed",
}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
W = 1280


def text_file(directory: Path, name: str, text: str) -> str:
    """drawtext reads its text from a file, so quotes, colons and percent signs need no escaping."""
    path = directory / name
    path.write_text(text)
    return str(path)


def segment(runs: Path, chip: int, kernel: str, label: str, result: dict, out: Path) -> None:
    scratch = out.parent
    kernel_wav = runs / f"chip{chip}" / f"{kernel}_kernel.wav"
    numpy_wav = runs / f"chip{chip}" / f"{kernel}_numpy.wav"
    title = f"{kernel}: {BLURB[kernel]}"
    sub = f"computed on {label}   worst difference from numpy {result['worst']:.1e}   run {result['run_ms']:.2f} ms"
    graph = (
        "[0:a]aresample=44100,asplit=3[w1][sp][out];"
        "[1:a]aresample=44100,showwaves=s={w}x150:mode=cline:colors=0xFF9E8A:rate=25,format=yuv420p[w2];"
        "[w1]showwaves=s={w}x150:mode=cline:colors=0x74C5DF:rate=25,format=yuv420p[w1v];"
        "[sp]showspectrum=s={w}x300:legend=0:color=intensity:scale=log:slide=scroll,format=yuv420p[spv];"
        "[w1v][w2][spv]vstack=inputs=3[v];"
        "[v]pad={w}:in_h+130:0:130:color=0x092221,"
        "drawtext=fontfile={font}:textfile={t}:x=24:y=18:fontsize=34:fontcolor=0xF1F8F8,"
        "drawtext=fontfile={font}:textfile={s}:x=24:y=68:fontsize=20:fontcolor=0x74C5DF,"
        "drawtext=fontfile={font}:textfile={k}:x=24:y=98:fontsize=16:fontcolor=0x9db9b6[vout]"
    ).format(w=W, font=FONT, t=text_file(scratch, f"{kernel}_t.txt", title), s=text_file(scratch, f"{kernel}_s.txt", sub),
             k=text_file(scratch, f"{kernel}_k.txt", "top: the chip   middle: numpy   bottom: the chip spectrogram"))
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(kernel_wav), "-i", str(numpy_wav), "-filter_complex", graph,
         "-map", "[vout]", "-map", "[out]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25", "-c:a", "aac", "-ar", "44100", "-shortest", str(out)],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", required=True, help="the --out directory that kernels/cycle.py wrote")
    parser.add_argument("--chip", type=int, default=0, help="which chip's saved audio to use (default 0)")
    parser.add_argument("--label", default="a Blackhole chip", help="what the caption calls the chip")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    runs = Path(args.runs)
    results = {}
    for line in (runs / "results.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("device") == args.chip:
            results[r["kernel"]] = r
    with tempfile.TemporaryDirectory() as tmp:
        parts = []
        for k in KERNELS:
            part = Path(tmp) / f"{k}.mp4"
            segment(runs, args.chip, k, args.label, results[k], part)
            parts.append(part)
        listing = Path(tmp) / "list.txt"
        listing.write_text("".join(f"file '{p}'\n" for p in parts))
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", args.out], check=True)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
