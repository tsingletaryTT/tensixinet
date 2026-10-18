# Sound design

## Ciat-Lonbarde ideas used

[CIRCUITS.md](CIRCUITS.md) lists what the instrument guides say and what the code does about each point.
In short:

* **Plumbutter 2:** sources are patched together and mix by current flow. Here each chip's share of the
  mix is its share of the total current, through a saturating stage, so piling on chips compresses the
  sound and never clips it. Patches also send one clock to several modules, so the gong, the shimmer's
  snap and the drone's ripple share one set of strike times.
* **Rolzer:** rolls of 3 to 6 stages on barres with E6 capacitor tempos. Even rolls swing and stay steady.
  Odd rolls are unstable. Capacitors are good to 20%, so no step lasts the same as another.
* **Sidrax:** seven oscillators that bounce between two bounds. Frequency is bounce over bounds, and the
  bounds can be other barres' oscillators.
* **Cocoquantus:** a five-petal Quantussy whose castles sample each other, through an 8-bit buffer.
* **Gongue, AVDog, Ultrasound, shift register and Snare** from the Plumbutter: a gong with a trigger
  response, a ripple after each strike, a snap, a pattern of steps and a noise snare.

## Signal to sound

| Hardware signal | Musical parameter | Range |
|---|---|---|
| `asic_temperature` | Pitch, linear in semitones | 25-100 C maps to C3-C5 |
| `aiclk` | Tempo | 400-1350 MHz maps to 60-180 BPM |
| `power` | Chip volume, exponential curve | 0-200 W maps to 0.3-1.0 |
| `current` | Share of the mix | each chip's fraction of the total |
| GDDR temperatures | Sidrax walls, lead shifts and inversions, shimmer beat rate | every channel the board reports |
| `ddr_status` | Rollz rolls of 3, 4, 5 and 6 sides, each on its own barre; status 0 grounds the roll | 4 fields |
| `voltage` | Drone detune: 0.75 V is in tune, 4 cents per 10 mV, at most 40 cents either way | 0.70-0.90 V |
| `pcie_usage` | Number of drone voices, and the snare's dust (15 pops a second per step) | 1-6 |
| `arcclk` | Sidrax pitch, 1.0 at 800 MHz | any |
| `axiclk` | Cocoquantus hold rate (1.0 at 960 MHz, a note is an eighth) and buffer clock (full rate at 1920 MHz, down to a quarter) | any |
| `eth_mask` | Four more Rollz rolls on slow barres, one per group of four ethernet links, as loud as the share of the group that is up. Also part of the chip identity | 16 links |
| `tensix_mask` | Part of the chip identity. The harvested columns (zero bits) are the steps the snare fires on | 14 columns |
| `heartbeat` | Where in the first blocking window the gong starts: counter mod 16, in sixteenths of half a strike interval | any |
| Pressure | Lead overtone (up to 60%), shimmer level (up to twice), Sidrax chaos, snare decay | 0-1, see below |
| `board_power`, `busy` | Turn weight, see "Trading solos" | |
| `corr_errs`, `uncorr_errs`, `thermal_trips` | Event sounds, see "Events" | counts |

The functions are in `tensixinet/music/mapping.py`. Readings outside a range hold the nearest end.

## The eight devices

| Device | Character | Reads |
|---|---|---|
| Rollz | Lopsided pulse clock plus a roll for each DDR channel and each group of ethernet links | temperature, aiclk, DDR status, ethernet links |
| Sidrax | Seven barres. A press plays a tone and a release plays a glitch | temperature, aiclk, GDDR temperatures, arcclk, pressure |
| Cocoquantus | A repeating, drifting melody on the patch's scale, through an 8-bit buffer | temperature, aiclk, axiclk, chip identity |
| Plumbutter drone | Drone of 1 to 6 C major voices that ripples after each gong strike | temperature, voltage, PCIe usage, aiclk, heartbeat, chip identity |
| Lead | Eight-note motif, shifted or inverted bar by bar | temperature, aiclk, GDDR temperatures, pressure, chip identity |
| Gong | Inharmonic strikes; the default tuning has partials 1, 2.76, 5.40, 8.45, each doubled and detuned 0.7 Hz | temperature, aiclk, heartbeat, chip identity |
| Ultrasonic shimmer | Two close tones in 12-20 kHz, with a snap on each gong strike | temperature, aiclk, GDDR temperatures, pressure, heartbeat, chip identity |
| MWTRS and Snare | A noise snare on the steps of a 14-step register that the harvested Tensix columns pick, over a bed of dust | aiclk, Tensix mask, pressure, PCIe usage |


`DEVICE_INPUTS` in `music/devices.py` is the exact list, and a test checks it field by field. The lead and
the shimmer read power and the throttle limits only to work out pressure.

## Strikes: one clock for the gong, the shimmer and the drone

A chip's clock is the lopsided pulse train that Rollz plays: long and short steps around the aiclk tempo. The
gong does not strike on a fixed grid. Its trigger response is a blocking window of two bars, times 1 to 1.375
by the chip's identity (one of seven steps of one sixteenth). A strike lands on the first clock pulse
after the window, so the strikes sit on the chip's own rhythm and different chips divide their clock
differently. The heartbeat sets how far into the first window the chip starts.

The same strike times drive two more devices, as a Plumbutter patch sends one clock to several modules:

* the shimmer snaps: its level goes from 70% to 100% and falls back with a 12 ms time constant
* the drone ripples: its level follows a damped swing, rectified, at 1.5 swings a beat and fading over two
  beats, then settles to 60%

## Rollz

The chip's own clock alternates long and short steps (25% swing) at the aiclk tempo. Each lane adds a roll with its own geometry and tempo:

| Lane | Roll | Barre | Tempo against the chip's own |
|---|---|---|---|
| DDR 0 to 3 | 3, 4, 5, 6 sides | 1, 2, 3, 4 | 1.45, 1.00, 0.67, 0.46 |
| Ethernet group 0 to 3 | 6, 5, 4, 3 sides | 3, 4, 5, 6 | 0.67, 0.46, 0.32, 0.21 |

Barre tempos are 192, 122, 84, 56, 39, 27 and 18 BPM, and barre 2 (84 BPM) plays at the chip's tempo. Every
step is jittered by up to 20%. An even roll holds that to half of it and swings. An odd roll magnifies it
2.5 times and spreads its steps unevenly over a cycle. No lane is locked to the bar. A DDR status of 0 grounds
its roll.

## Sidrax

Seven barres tuned to the first seven degrees of the patch's scale, starting at the chip's pitch times
arcclk / 800 MHz. Each barre has fixed walls, its own GDDR channel for the upper one and the next channel for the
lower one: `-(0.2 + 0.7 x norm_temp)` and `0.2 + 0.7 x norm_temp`.

* **Press:** the tone bounces between the fixed walls.
* **Release:** the glitch bounces between two other barres' oscillators, the pair the guide's table gives
  for switch 3 (`BOUNDS_TABLE` in `music/sidrax.py`). Its frequency is the tone's times the fixed window over
  the current window, between a quarter and eight times. Where the upper wall drops below the lower, it rides the
  lower wall.
* **Chaos:** pressure, plus (in live mode) how fast the chip's temperature and power are moving. The rates add up to
  1 for 1 C/s or 25 W/s, the faster of the two, and the sum stops at 1. Each barre's frequency is swung by the
  oscillator two barres along, by up to 200% at full chaos. Offline there are no rates, so chaos is the pressure.
* **Gestures:** barre b is pressed every four steps of Rolzer barre b and released half a period later. Each
  envelope peaks on its gesture and falls with a time constant of 0.15 of a period.

## Cocoquantus

Five triangle oscillators (petals) at ratios 1, 3/2, 5/4, 4/3 and 5/3, each nudged by up to 3% from the seed.
Petal 0 reaches a peak or a valley once per note. Its castle samples a leader petal's triangle at each one and holds the value. The
leader is 2, 3, 4 or 1 places round the ring, picked by the chip's identity. The held value picks a note of the
patch's scale over two octaves. Low prime ratios make the notes repeat, and the nudge makes the pattern drift.
The voice is a sine rounded to 8 bits, held for longer at a lower AXI clock.

## MWTRS and Snare

A 14-step register, one step per Tensix column, stepped at eighth notes. A step fires where its column is
harvested (the mask bit is 0). The live chips each have two harvested columns, so a chip fires two steps in
fourteen and a chip with every column enabled is silent. A hit is white noise rolled off above 4 kHz, with an
envelope that falls to a third in 0.75 s at no pressure and 75 ms at full pressure. Under it, dust pops 15 times
a second for each step of PCIe usage.

## Pressure

How close a chip is to a limit, from 0 to 1, whichever of two ramps is higher:

* ASIC temperature from 40 C up to the chip's `THM_LIMIT_THROTTLE` (90 C on a Blackhole p300)
* power from a quarter of the chip's TDP limit up to the limit (125 W)

The lead adds an octave overtone at 60% of the note at full pressure. The shimmer doubles its level. Each
chip's turn weight also rises with pressure.

## Chip identity

Harvesting leaves each chip with a different set of Tensix columns, and the ethernet links that are up
differ too. Both masks are hashed to one number per chip. It transposes the chip's lead phrases by 0, 1 or 2
scale degrees and detunes its gong by -10 to +10 cents, so two chips playing the same device do not play it
alike. Only three lead offsets exist, so two chips share one about a third of the time.

## Events

A healthy chip reports zero for all three counters and adds nothing. Otherwise:

| Reading | Sound | Cap |
|---|---|---|
| Correctable GDDR errors | A short high tick per error | 40 |
| Uncorrectable GDDR errors | A low breath of noise per error | 3 |
| Thermal trips | A falling sweep from 2 kHz to 200 Hz per trip | 3 |

Events are added to the chip's signal after the turn gains, so they sound whoever has the floor. Use
`--mock-chips N --mock-faults` to hear them without a failing chip.

## Patches: a different set of choices every run

A patch holds the choices that used to be fixed constants. Two sources decide it.

**The system decides two things**, so the sound follows the machine:

| Choice | Decided by | Rule |
|---|---|---|
| Scale | GDDR spread (hottest minus coolest channel, all chips) | Each 3 C moves one scale along: major pentatonic, minor pentatonic, hirajoshi, dorian, whole tone, phrygian dominant. Even memory plays the first |
| Gong tuning | Average GDDR temperature divided by average ASIC temperature | Ratio times 10, rounded, picks bell, tam-tam or singing bowl |

**A seed decides the rest:** the lead motif (eight scale degrees from 0, steps of at most 2), the gong's
beat (0.4 to 1.2 Hz), the shimmer band (low edge 10 to 14 kHz, 4 to 7 kHz wide), the turn length (1 to 4
bars) and how far non-soloists drop (25% to 45%).

The seed is a hash of every reading on every chip plus entropy (the clock), so no two runs match. Each run
writes its patch to `patch_<time>.json` beside its log and prints every choice with its reason.
Replay one with `tensixinet --patch FILE`, or fix the seed with `--patch-seed N`. The built-in
`DEFAULT_PATCH` holds the constants from before patches existed (`tensixinet/music/patch.py`).

## Dealing devices to chips

There are always at least eight slots, one per device. Slot `s` goes to chip `s % chips`. One chip
plays all eight. Three chips play eight slots between them. Eight or more chips play one each.

## Trading solos

A turn lasts `bars_per_turn` bars (default 2) at the average tempo. In each turn one slot is featured
and the others drop to the patch's floor (default 35%) with 0.25 s fades. The soloist is drawn in proportion to its chip's turn weight,
never repeats back to back, and a quiet chip still gets turns.

The turn weight is the chip's power, times its board power against the average board, times 1 plus 2 per unit
of pressure, times 1 plus 0.5 for each process using the chip (up to three). A snapshot with no process list counts
none, and `tt-smi` itself is never counted, because it lists itself on every chip. `music/turns.py` draws from a
golden-ratio sequence, so the same chips always give the same order.

## The lead

Each bar reads the next GDDR channel and compares it with the average channel. Hotter than average
shifts the motif up one scale degree per degree C, at most four. Cooler inverts it around its first
note. Within 1 C it repeats. Even memory temperatures make the lead loop, and uneven ones make it
wander.

## Loudness

`render_chorus` keeps power as volume, so a quiet box peaks near 0.2 and a loaded one near 0.4. `run` then
multiplies the whole render by one number, `music/loudness.py`: the gain that brings its RMS to -18 dBFS, held
down so the peak stays under -1 dBFS and the gain stays under +20 dB. Chips keep their level against each other and
the swells and gaps survive, because one gain moves all of them. The log prints the gain. `--no-normalise` skips it.

## Ultrasonic content

The shimmer sits between 12 and 20 kHz and shrinks to fit below 0.45 of the sample rate. At 44.1 kHz
the top is 19.8 kHz. Use `--sample-rate 96000` for more headroom. Synth partials above Nyquist are
dropped so high voices never alias into the audible band.

## Live mode

`--live` plays the chorus one turn at a time (the patch's bars at the mean tempo, held between 2 and 10 s). Before
each turn the chips are read again and compared with the last readings:

* temperature and power get a rate in units per second, over at least 2 s, which feeds Sidrax's chaos;
* the memory error and thermal trip counters become what is new, so an error plays once;
* the soloist is drawn from the chips' weights as they are now;
* the turn is rendered with its own seeds and cross-faded for 0.4 s onto the end of the last.

Each turn restarts the devices' clocks, so a gong that rings past the end of its turn fades out instead. The level
follows the music with a running gain toward -18 dBFS RMS. See `DECISIONS.md` for why it is built this way.
