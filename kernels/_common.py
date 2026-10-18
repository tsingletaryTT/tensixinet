# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""What every kernel script shares: finding the package, opening a device safely, moving arrays to and from tiles,
and reporting one JSON line. Run kernels through `tt-lang-sim` or under `gozer run`; see README.md."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import ttnn  # noqa: E402

from tensixinet.kernels.packing import unpack, within_tolerance  # noqa: E402

TILE = 32


def is_simulator() -> bool:
    return "sim" in getattr(ttnn, "__name__", "")


def lease_guard() -> Optional[int]:
    """None when it is safe to open a device, else the exit code to stop with. The simulator is always safe. A real
    device needs a gozer lease, which `gozer run` announces by setting TT_VISIBLE_DEVICES."""
    if is_simulator() or os.environ.get("TT_VISIBLE_DEVICES"):
        return None
    print("refusing to open a real device without a lease: run this under `gozer run --chips 1 ...`", file=sys.stderr)
    return 3


def to_device(array: np.ndarray, device):
    """A float32 tiled tensor in DRAM."""
    return ttnn.from_torch(
        torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def from_device(tensor, n: int) -> np.ndarray:
    return unpack(ttnn.to_torch(tensor).float().numpy(), n)


def timed(op: Callable, *tensors) -> float:
    """Milliseconds for one call of a kernel that has already run once (the first call compiles it)."""
    start = time.perf_counter()
    op(*tensors)
    return (time.perf_counter() - start) * 1000.0


def report(name: str, got: np.ndarray, expected: np.ndarray, args: argparse.Namespace, sample_rate: int, run_ms: float) -> int:
    """Compare, save the audio when asked, and print the one JSON line. 0 when within `args.atol`."""
    ok, worst = within_tolerance(got, expected, args.atol)
    if args.save:
        out = Path(args.save)
        out.mkdir(parents=True, exist_ok=True)
        sf.write(str(out / f"{name}_kernel.wav"), got, sample_rate, subtype="PCM_16")
        sf.write(str(out / f"{name}_numpy.wav"), expected, sample_rate, subtype="PCM_16")
    print(json.dumps({"kernel": name, "samples": len(expected), "sample_rate": sample_rate, "worst": worst, "atol": args.atol,
                      "ok": ok, "run_ms": round(run_ms, 3), "device": args.device, "simulator": is_simulator()}))
    return 0 if ok else 1


def run_check(
    name: str,
    op: Callable,
    inputs: Sequence[np.ndarray],
    expected: np.ndarray,
    args: argparse.Namespace,
    sample_rate: int,
) -> int:
    """Run `op(*inputs, y)` on a device (once to compile it, once timed), compare with `expected` and report."""
    n = len(expected)
    device = ttnn.open_device(device_id=args.device)
    try:
        tensors = [to_device(a, device) for a in inputs]
        y = to_device(np.zeros_like(inputs[0]), device)
        op(*tensors, y)
        run_ms = timed(op, *tensors, y)
        got = from_device(y, n)
    finally:
        ttnn.close_device(device)
    return report(name, got, expected, args, sample_rate, run_ms)


def parse_args(default_samples: int, default_atol: float) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check a kernel against the numpy device it ports.")
    parser.add_argument("--samples", type=int, default=default_samples, help=f"how many samples to compute (default {default_samples})")
    parser.add_argument("--atol", type=float, default=default_atol, help="largest allowed difference from the numpy device")
    parser.add_argument("--device", type=int, default=0, help="which of the leased chips to open (0 is the first)")
    parser.add_argument("--save", metavar="DIR", help="write what the kernel computed and the numpy reference there as 16-bit WAV files")
    return parser.parse_args()
