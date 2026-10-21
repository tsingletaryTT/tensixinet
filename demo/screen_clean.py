# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Is the screen clean enough to record? Reads a 1920x1080 screenshot and exits 0 when no dialog or taskbar is showing, 1 when one is.

    python3 screen_clean.py screenshot.png

A modal dialog (OBS throws a "Plugin Load Error" one) is a large pale grey rectangle over the middle of the screen, so the check is how much of that
middle is pale. tt-bio-demo's own check probed one row of pixels there, and in the quad view the caption text of a cell sits on that row, so a clean
screen was refused. The taskbar check is the same as theirs: bright pixels along the bottom row, where the booth is dark."""
import sys

from PIL import Image

DIALOG_BOX = (830, 400, 1120, 540)  # where the dialog lands
PALE = 200  # a pixel this bright, in luminance, is dialog-coloured
DIALOG_SHARE = 0.30  # a dialog fills most of the box; text fills a few percent
TASKBAR_ROW = 1058
TASKBAR_BRIGHT = 60
TASKBAR_SAMPLES_MAX = 6  # of 95 probes along the row


def measure(im: Image.Image) -> tuple:
    """(share of the dialog box that is pale, taskbar probes that are bright out of 95)."""
    box = im.convert("L").crop(DIALOG_BOX)
    pixels = list(box.getdata())
    pale = sum(1 for p in pixels if p > PALE) / len(pixels)
    taskbar = sum(1 for p in (im.convert("RGB").getpixel((x, TASKBAR_ROW)) for x in range(0, 1900, 20)) if all(c > TASKBAR_BRIGHT for c in p))
    return pale, taskbar


def main(path: str) -> int:
    pale, taskbar = measure(Image.open(path))
    print(f"  precheck: dialog {pale:.0%} pale (limit {DIALOG_SHARE:.0%}), taskbar {taskbar}/95 (limit {TASKBAR_SAMPLES_MAX})")
    if pale > DIALOG_SHARE or taskbar > TASKBAR_SAMPLES_MAX:
        print("  REFUSING: chrome is still on screen")
        return 1
    print("  screen is clean")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
