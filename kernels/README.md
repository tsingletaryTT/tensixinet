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
| `ttchoir.py` | `ttchoir` (the choir's voiced part, both the vowel choir and the syllable choir) | `sum_j a_j * sin(p_j)` over eight harmonics, built in four stages of two (four working registers); run once per eight harmonics, so up to five runs for forty | each harmonic's phase, `(k + 1)` times the voice's phase reduced to one turn in float64, and its level interpolated from the 200 Hz frame table (`choir_groups`); adds the runs; the syllable plan, consonants, howl and crush through the device's own code (`choir_voice` with a `sines` callable) |

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

## The choir on the chips (2026-10-31)

`ttchoir.py` ran on chips 0 and 1 (board 0, one `gozer` lease; board 1 was reserved for another agent and was not used). `cycle.py` checks it
twice, the vowel choir and the syllable choir (`--voice syllable`, saved as `ttchoir_syllable`). Raw results:
`docs/hardware/board0_ttchoir_results.jsonl` and `docs/hardware/board0_ttchoir_timing.jsonl`.

| Check | Chip 0 | Chip 1 | Simulator |
|---|---|---|---|
| Vowel choir, 3 s at 16 kHz, worst difference | 1.5e-7 | 1.5e-7 | 1.5e-7 |
| Syllable choir, 3 s at 16 kHz, worst difference | 1.2e-7 | 1.2e-7 | 1.2e-7 |
| Kernel time for those 3 s (four or five runs) | 2.1 to 2.5 ms | 2.1 to 2.6 ms | |
| 30 s at 44.1 kHz through the worker's chunked path, vowel | 0.87 s (CPU 0.65 s) | 0.88 s (CPU 0.62 s) | |
| 30 s at 44.1 kHz, syllable choir | 0.70 s (CPU 0.55 s) | 0.68 s (CPU 0.53 s) | |

The 30 s renders matched the CPU choir to 1.5e-7 at the worst, correlation 1.0000000, also with a howl and with the crush at full depth.
**The chip path is slower than the CPU**, about 1.3 times: five runs of sixteen input tiles move about 420 MB to the chip for 30 s of
voice, against well under a second of numpy. The 30 s timings drive `worker.py`'s own code (`Board`, `chip_compute`, `run_chunked` with its
448-tile chunks) in one process, so they leave out the pipe between `tensixinet` and the worker.

Two things were found on the way:

* **The chip's multiplies were exact here.** `ttavdog` and `ttcoco` showed differences of 1e-3; the choir's `a * sin(p)` sum showed
  1.5e-7 on the chip, the same as the simulator. `--atol` is 2e-3 anyway, as wide as the loosest kernel, in case another toolchain rounds
  the multiplies as before.
* **The toolchain on this box had moved.** On 2026-10-31 `ttchoir` and `ttcoco` (which ran here on 2026-10-18) both failed to compile with
  `fatal error: tools/profiler/kernel_profiler.hpp: No such file or directory`. tt-metal commit `27cad27903e` dropped `<root>/tt_metal`
  from the JIT's include path, and TT-Lang (1.1.6 in the shared venv, and tt-orchard's pinned 1.1.10 pair) still generates that include.
  The choir ran with tt-orchard's pair venv (`~/.cache/tt-orchard/ttlang-pair/venv` and its `orchard-pair-exports.sh`) and a private
  runtime root: a folder of symlinks to `~/tt-metal` with one added file, `tools/profiler/kernel_profiler.hpp`, that includes
  `api/debug/kernel_profiler.hpp`, passed as `TT_METAL_RUNTIME_ROOT` to that process only. Nothing shared was changed. With that, `ttchoir`
  compiles and runs, but `ttcoco` fails on `add_tiles_init` and `binary_op_init_common`, which this tt-metal no longer declares. The other
  four old kernels were not tried. `worker.py` compiles every kernel at start, so `--kernels chips` cannot start on this box until TT-Lang and
  tt-metal agree again. None of the old kernels changed.

## The soundtrack on the chips (`tensixinet --live --kernels chips`)

`--kernels` runs the seven kernels' devices (the gong, shimmer, drone ripple, AVDog, the ttcoco mix, the choir's harmonics and the mixer) on chips for the whole
of a live or offline run, with the rest of the band on the CPU. Each chip's devices run on that chip, so on four chips all four work.
On the four chips of this box the audio matched the CPU render to 0.0034 at the worst (correlation 1.00000, 12 s, `--patch-seed 4`).

How it is built, and what it cost to find out:

* `tensixinet/media/chip.py` (the client) starts `gozer run --chips N ... kernels/pool.py` and talks to it over a pipe. The package never
  imports `ttnn`. `kernels/pool.py` starts one `kernels/worker.py` per board; a worker keeps the kernels compiled and runs each request
  in fixed-size chunks (448 tiles, 10.4 s at 44.1 kHz) so each kernel compiles once. `tensixinet/kernels/backend.py` packs the inputs the
  way the kernel scripts do, calls the worker, and falls back to the CPU, with one log line, if the worker fails.
* **A board's two chips cannot be worked at once.** With both devices open in one process the first kernel launch hung (an
  experiment, under `timeout`). A second process on the same board waited until the first had closed it. Switching from one chip to
  the other inside a process costs about 0.1 s once the kernels are compiled. So a worker keeps one chip open and switches when a
  request names the other.
* **A worker must be shown exactly its board.** Four workers that each saw all four chips waited on each other's locks
  ("CHIP_IN_USE_0_PCIe"). One worker shown one chip of a two-chip board made tt-metal call the cluster CUSTOM and refuse to start.
  One worker per board, shown that board's two chips, runs in parallel with the other board.
* **Closing takes about 40 s.** `gozer run` resets and re-initialises the chips before it releases the lease. A first version gave up
  after 30 s and killed `gozer run`, which left its worker holding the chips and the lease to be reaped as "owning process is dead".
  The client now waits up to 180 s, and if it must end a worker it ends the whole process group.
* **Do not run `gozer status` while a lease is being released.** It probes the devices and can make the release refuse with
  "device still open", which leaves the lease stale until `gozer reconcile`. A live run's own `tt-smi -s` reads, made by another
  process, can do the same.
* A request that gozer queues (exit 10) leaves a ticket behind; the client cancels its own ticket.
* A live segment renders in about 0.5 s on the chips against 0.4 s on the CPU: the chips are not faster for this work, the transfer
  costs more than the 0.2 ms of arithmetic. The point is that the sound is made by the kernels.

What the chip taught us that the simulator did not:

* **The compiler refuses an input buffer that feeds both `sin` (a vector-unit op) and an addition (a matrix-unit op).** `ttgong` added
  `a + b` and also took `sin a`. The host now gives each detuned copy's phase as a tile of its own (sixteen input tiles).
* **Four working registers.** `ttavdog`'s one long expression ("insufficient DST registers"), and `ttgong`'s sum of five terms, are built
  in stages that pass partial results through extra dataflow buffers.
* **Multiplies are looser than the transcendentals.** `sin` and `exp` match numpy to 1e-7, but `tanh` is off by 3e-4 and the
  multiplies in `ttavdog` by 2e-3 (about 10 bits of mantissa). That is far below hearing and is the reason for the two looser tolerances.
* One timed run (the first call compiles) takes 0.2 to 0.7 ms for one to ten seconds of audio, so a single core is hundreds of
  times faster than real time for any of these devices.

## In the simulator, for a machine with no chip (`--kernels sim`)

`tensixinet --live --kernels sim` runs the same worker and pool as `--kernels chips`, but each worker is started by `tt-lang-sim` and
no lease is taken. The pool starts one worker per pretend board and the simulator answers for two chips on each. Starting a worker takes
up to two minutes (the simulator imports torch), so `SIM_START_TIMEOUT_S` is 120 s. `find_simulator` looks at `TT_LANG_SIM`, then the
path, then beside the running Python. `tests/media/test_chip_sim.py` (marked `sim`) runs every kernel through the pool on four pretend
chips and compares with the host formula to 1e-4. It skips unless `TT_LANG_SIM` names the simulator.

The Hugging Face Space (`space/`, built by `space/assemble.sh`) uses this mode with synthetic chips. It has no Tenstorrent hardware and
says so on its page. Nothing has been published.

