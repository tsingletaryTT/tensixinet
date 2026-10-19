# Kernels

The synth devices as TT-Lang kernels, so the chips can play the sound as well as be heard through it. The numpy devices in
`tensixinet/music/` stay as the reference: each kernel is checked against the device it ports, sample by sample.

| Kernel | Ports | What the chip does | What the host does |
|---|---|---|---|
| `ttshock.py` | `shock_wave` (the ttdrone's ripple after each gong strike) | `live * abs(sin(x)) * exp(neg_e)` for every sample | phase and decay exponent per sample, in float64; packs samples into 32 x 32 tiles |
| `ttshimmer.py` | `ultrasonic_shimmer` (two tones that snap on every strike) | `(sin p1 + sin p2) * (a0 + a1 * exp(e))` | each tone's phase reduced to one turn in float64, the snap exponent, the two level constants |
| `ttgong.py` | `gong` (one ringing strike, up to five partials each with a detuned copy) | `att * sum_p exp(e_p) * (sin a_p + sin c_p)`, built in stages (four working registers); run once per overlapping strike (three) | phases with the pitch sag in closed form, each partial's own tile and its detuned copy's own phase tile, decay exponents with the amplitudes folded in, the attack ramp; adds the runs and applies `0.6 * tanh` |
| `ttavdog.py` | `ttavdog` (the AVDog's white and red voices) | `relu(u) * white_wave + relu(-u) * red_wave`, with `u = sin(pu) * exp(e)` and `min`-skewed triangles with swapped slopes | the dogtone's phase, the slopes, the undulation's phase and decay, and four constants for the level and red gain (the chip takes tensors, not numbers) |
| `ttmix.py` | `current_mix` (the chips' mixer) | `tanh(a + b + c + d)` | each chip's weight and `1 / sqrt(chips)`, folded into its signal; up to four chips |
| `ttcoco.py` | `ttcoco` (the Cocoquantus's delay chips, as eight parallel echo taps) | `tanh(a0 + a1 + ... + a7)`, built in stages (four working registers) | each tap's delay, decay and bit-crunch (`echo_tap`, a truncated sum of repeats, like the gong's ring-down) and the device's own headroom term |

## How a kernel is written here

* One file per kernel, named `tt<device-or-part>.py`, with a `@ttl.operation` and a `main()` that builds inputs from the
  numpy device's own helpers, runs the operation, unpacks the result and prints one JSON line: the worst difference from the
  numpy device and whether it is within `--atol`. Exit 0 means within tolerance.
* Block arithmetic takes tensors, not Python numbers (the simulator refuses `blk * 0.5`), so constants such as a level are packed
  into tiles of their own. `ttl.copy` must be a bare call or `tx = ttl.copy(...)`, not a list of copies.
* **Two things the simulator does not check, and a real chip does.** One input buffer cannot feed both `sin` (or another vector-unit
  op) and a plain addition (a matrix-unit op) in the same kernel; give the two kinds of consumer separate tiles, even if that means
  computing a sum like `a + beat` on the host instead of the chip. And a kernel has four working (DST) registers; an expression with
  more terms than that fails to compile ("insufficient DST registers: spilling not yet implemented"), so build it in stages, each
  stage's result passed to the next through its own dataflow buffer (`ttgong.py` and `ttavdog.py` do this).
* The chip does the per-sample transcendental work (`sin`, `cos`, `exp`, `tanh`, `abs` and the arithmetic around them).
  The host does the cheap multiplications that make each sample's argument, in float64, because a phase of 40 radians in
  bfloat16 is off by a quarter of a radian. Tensors are float32.
* The pure-numpy side (packing, and the host/chip split of each formula) is in `tensixinet/kernels/` and is tested in the
  ordinary suite. The kernel files import `ttl`, so they live here, outside the package, and the hygiene test does not
  let `tensixinet/` or `tests/` import `ttl` or `ttnn`.

## Run in the simulator (no chip, no lease)

The simulator is Tenstorrent's `tt-lang-sim`. Install it in a fresh venv that has no `ttnn`, because the `tt-lang-sim` that
comes with the full TT-Lang install can reach the real `ttnn`:

```bash
python3 -m venv simvenv && simvenv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
simvenv/bin/pip install tt-lang-sim numpy soundfile pytest
simvenv/bin/tt-lang-sim kernels/ttshock.py
TT_LANG_SIM=$PWD/simvenv/bin/tt-lang-sim make kernels-sim      # every kernel
TT_LANG_SIM=$PWD/simvenv/bin/tt-lang-sim python3 -m pytest tests/test_kernels_sim.py
```

## Run on a chip (needs a lease, and a turn)

The box is shared. A kernel run opens a chip, so it goes through gozer, and the script refuses to open a real device unless
`gozer run` set `TT_VISIBLE_DEVICES`:

```bash
gozer run --chips 1 --who "claude:tensixinet" --reason "ttshock kernel check" -- python3 kernels/ttshock.py --atol 5e-3
```

Ask for one chip at a time and for short slots. If the chips are held, `gozer` queues the request: do simulator work and
`gozer wait` on the ticket. Never reset or release a chip you do not hold. tt-orchard shares the box, and has asked that a
request not be queued ahead of its own run.

**The simulator's error messages are broken in the PyPI wheel.** On any error `tt-lang-sim` tries to load `ttl/ttl/diagnostics.py`,
which the wheel does not contain, and shows `FileNotFoundError` instead of the real message. To see the real error, copy
`diagnostics.py` from a full TT-Lang install (`site-packages/ttl/diagnostics.py`) to `site-packages/ttl/ttl/diagnostics.py` in the
simulator venv. This is a bug in `tt-lang-sim` 1.1.6, not in these kernels.

## What happened on the chips (2026-10-18)

All six kernels ran on all four Blackhole chips (two p300c boards, one `gozer` lease per board) and matched the numpy devices.
`python3 kernels/cycle.py --out DIR` does the whole cycle and `--save` keeps what each chip computed as WAV files beside the
numpy reference. Raw results are in `docs/hardware/`.

| Kernel | Worst difference on a chip | Same in the simulator | Time for one timed run | Tolerance now |
|---|---|---|---|---|
| `ttshock` | 3.1e-7 | 3.1e-7 | 0.2 ms | 2e-4 |
| `ttshimmer` | 1.5e-7 | 1.5e-7 | 0.2 ms | 2e-3 |
| `ttgong` | 2.0e-7 | 1.4e-7 | 0.5 to 0.7 ms (three strikes) | 2e-4 |
| `ttavdog` | 2.3e-3 | 1.2e-7 | 0.2 to 0.4 ms | 5e-3 |
| `ttmix` | 2.8e-4 | 6e-8 | 0.2 ms | 1e-3 |
| `ttcoco` | 8.8e-4 | 1.2e-7 | 0.2 to 0.4 ms | 1e-3 |

What the chip taught us that the simulator did not:

* **The compiler refuses an input buffer that feeds both `sin` (a vector-unit op) and an addition (a matrix-unit op).** `ttgong` added
  `a + b` and also took `sin a`. The host now gives each detuned copy's phase as a tile of its own (sixteen input tiles).
* **Four working registers.** `ttavdog`'s one long expression ("insufficient DST registers"), and `ttgong`'s sum of five terms, are built
  in stages that pass partial results through extra dataflow buffers.
* **Multiplies are looser than the transcendentals.** `sin` and `exp` match numpy to 1e-7, but `tanh` is off by 3e-4 and the
  multiplies in `ttavdog` by 2e-3 (about 10 bits of mantissa). That is far below hearing and is the reason for the two looser tolerances.
* One timed run (the first call compiles) takes 0.2 to 0.7 ms for one to ten seconds of audio, so a single core is hundreds of
  times faster than real time for any of these devices.
