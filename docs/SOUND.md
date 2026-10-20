# Sound design

## Ciat-Lonbarde ideas used

The devices are inspired by Ciat-Lonbarde instruments and are not recreations of them, so ours carry a `tt` prefix.

[CIRCUITS.md](CIRCUITS.md) lists what the instrument guides say and what the code does about each point.
In short:

* **Plumbutter 2:** sources are patched together and mix by current flow. Here each chip's share of the
  mix is its share of the total current, through a saturating stage, so piling on chips compresses the
  sound and never clips it. Patches also send one clock to several modules, so the gong, the shimmer's
  snap and the drone's ripple share one set of strike times.
* **ttrolzer:** rolls of 3 to 6 stages on barres with E6 capacitor tempos. Even rolls swing and stay steady.
  Odd rolls are unstable. Capacitors are good to 20%, so no step lasts the same as another.
* **ttrax:** seven oscillators that bounce between two bounds. Frequency is bounce over bounds, and the
  bounds can be other barres' oscillators.
* **ttquantus:** a five-petal ttquantussy whose castles sample each other, through an 8-bit buffer.
* **Gongue, AVDog, Ultrasound, shift register and Snare** from the Plumbutter: a gong with a trigger
  response, a ripple after each strike, a snap, a pattern of steps and a noise snare.

## Signal to sound

| Hardware signal | Musical parameter | Range |
|---|---|---|
| `asic_temperature` | Pitch, linear in semitones | 25-100 C maps to C3-C5 |
| `aiclk` | Tempo | 400-1350 MHz maps to 60-180 BPM |
| `power` | Chip volume, exponential curve | 0-200 W maps to 0.3-1.0 |
| `current` | Share of the mix | each chip's fraction of the total |
| GDDR temperatures | ttrax walls, lead shifts and inversions, shimmer beat rate | every channel the board reports |
| `ddr_status` | ttrollz rolls of 3, 4, 5 and 6 sides, each on its own barre; status 0 grounds the roll | 4 fields |
| `voltage` | Drone detune: 0.75 V is in tune, 4 cents per 10 mV, at most 40 cents either way | 0.70-0.90 V |
| `pcie_usage` | Number of drone voices, and the snare's dust (15 pops a second per step) | 1-6 |
| `arcclk` | ttrax pitch, 1.0 at 800 MHz | any |
| `axiclk` | ttquantus hold rate (1.0 at 960 MHz, a note is an eighth) and buffer clock (full rate at 1920 MHz, down to a quarter) | any |
| `eth_mask` | Four more ttrollz rolls on slow barres, one per group of four ethernet links, as loud as the share of the group that is up. Also part of the chip identity | 16 links |
| `tensix_mask` | Part of the chip identity. The harvested columns (zero bits) are the steps the snare fires on | 14 columns |
| `heartbeat` | Where in the first blocking window the gong starts: counter mod 16, in sixteenths of half a strike interval | any |
| Pressure | Lead overtone (up to 60%), shimmer level (up to twice), ttrax chaos, snare decay | 0-1, see below |
| `board_power`, `busy` | Turn weight, see "Trading solos" | |
| `corr_errs`, `uncorr_errs`, `thermal_trips` | Event sounds, see "Events" | counts |

The functions are in `tensixinet/music/mapping.py`. Readings outside a range hold the nearest end.

## The eleven devices

| Device | Character | Reads |
|---|---|---|
| ttrollz | Lopsided pulse clock plus a roll for each DDR channel and each group of ethernet links | temperature, aiclk, DDR status, ethernet links |
| ttrax | 4, 7 or 14 barres (Tetrax, Sidrax or Quatrax). A press plays a tone and a release plays a glitch | temperature, aiclk, GDDR temperatures, arcclk, pressure |
| ttquantus | A repeating, drifting melody on the patch's scale, through an 8-bit buffer | temperature, aiclk, axiclk, chip identity |
| ttdrone | Drone of 1 to 6 C major voices that ripples after each gong strike | temperature, voltage, PCIe usage, aiclk, heartbeat, chip identity |
| Lead | Eight-note motif, shifted or inverted bar by bar | temperature, aiclk, GDDR temperatures, pressure, chip identity |
| Gong | Inharmonic strikes; the default tuning has partials 1, 2.76, 5.40, 8.45, each doubled and detuned 0.7 Hz | temperature, aiclk, heartbeat, chip identity |
| Ultrasonic shimmer | Two close tones in 12-20 kHz, with a snap on each gong strike | temperature, aiclk, GDDR temperatures, pressure, heartbeat, chip identity |
| ttavdog | A drone oscillator opened by a ringing filter: a white voice on one half of each swing, a red voice on the other | aiclk, temperature, heartbeat, pressure, chip identity |
| ttchoir | A singer: a buzz through formant resonances, sung on vowels at the chip's pitch | temperature, aiclk, GDDR temperatures, pressure, chip identity |
| ttsteam and Snare | A noise snare on the steps of a 14-step register that the harvested Tensix columns pick, over a bed of dust | aiclk, Tensix mask, pressure, PCIe usage |
| ttcoco | The Cocoquantus's own delay chips: eight parallel echo taps, one per GDDR channel, each independently timed and bit-crunched, echoing the ttquantussy's own tone | aiclk, axiclk, GDDR temperatures, pressure, chip identity |


`DEVICE_INPUTS` in `music/devices.py` is the exact list, and a test checks it field by field. The lead and
the shimmer read power and the throttle limits only to work out pressure.

## Strikes: one trigger for the gong, the shimmer, the drone and the AVDog

The gong does not strike on a fixed grid. Its trigger response is a blocking window of two bars, times 1 to 1.375
by the chip's identity (one of seven steps of one sixteenth). A strike lands on the first pulse of its trigger
after the window, so different chips divide their trigger differently. The heartbeat sets how far into the first
window the chip starts.

The trigger is one of two things, picked by the chip's identity and the patch and kept for the whole run. Three times in
four the gong is patched to one of the chip's rolls, as a Gongue is patched to a roll's orange in the guide, and
strikes on that roll's own pulses, which wander off any grid. The other times, and for a chip with no rolls up, it
follows the chip's clock: the lopsided pulse train that ttrollz plays, long and short steps around the aiclk tempo.
The roll can be any of the chip's rolls, heard or not (see ttrollz below), and the pulses it follows are the ones
ttrollz plays, because both read the same drift for the same segment of a live run.

The same strike times drive three more devices, as a Plumbutter patch sends one trigger to several modules:

* the shimmer snaps: its level goes from 70% to 100% and falls back with a 12 ms time constant
* the drone ripples: its level follows a damped swing, rectified, at 1.5 swings a beat and fading over two
  beats, then settles to 60%
* the AVDog pings its filter (see ttavdog)

## ttrollz

The chip's own clock alternates long and short steps (25% swing) at the aiclk tempo. Each lane that is up brings a
barre of rolls, as the Rolzer has: a triangle, a square, a pentagon and two hexagons (3, 4, 5, 6 and 6 sides), the
second hexagon pulsing out of phase with the first. A chip with everything up runs 40 rolls (the real Rolzer has 35,
five on each of seven barres). Only one roll of each barre is heard, with the geometry and tempo below; the rest are
silent, but they are linked into the network and a gong can be patched to any of them.

| Lane | The roll that is heard | Barre | Tempo against the chip's own | Rolls the lane brings |
|---|---|---|---|---|
| DDR 0 to 3 | 3, 4, 5, 6 sides | 1, 2, 3, 4 | 1.45, 1.00, 0.67, 0.46 | all five |
| Ethernet group 0 to 3 | 6, 5, 4, 3 sides | 3, 4, 5, 6 | 0.67, 0.46, 0.32, 0.21 | as many as its share of links up (rounded up), and always its own |

Barre tempos are 192, 122, 84, 56, 39, 27 and 18 BPM, and barre 2 (84 BPM) plays at the chip's tempo. The lanes
run together as coupled circuits (`ttrolzer.roll_ensemble`), not as independent pulse trains:

* **Capacitors that stay.** Each roll's capacitors differ by up to 20% (10% for an even roll) and keep that mismatch,
  so a 4-roll trots the same way every cycle. The mismatch comes from the patch and the chip's identity, so it does
  not change when live mode renders the next segment. It is rescaled so it never moves the barre's mean tempo.
* **Geometry.** A 4-roll alternates long and short (25% swing). A 6-roll swings only 5% and carries two threes
  interleaved (26%), so it feels in triplets. A 3-roll and a 5-roll spread their steps unevenly over a cycle of 3 or 5.
* **Wander.** Every step has 1% loose timing. Each capacitor drifts, with a memory, by 0.5% a step on a cool chip up
  to 5% at 90 C and above (a random walk pulled back by 10% a step). An odd roll's pattern also slides round its
  cycle by a random walk of 12% of a cycle per cycle, so it never comes back round in the same phase.
* **Marked cycle.** The first step of each roll's cycle also plays an octave-up blip at 0.6 of the lane's level, so a
  3 or a 5 can be heard.
* **Browns as Sandrodes.** Links join pairs of the chip's rolls: two to four, plus one for every ten rolls and one for
  every five (a 40-roll chip gets 6 to 12). They are drawn from the patch seed and the chip's identity so a chip is wired
  the same way for the whole run. At least half of them end at a roll that is heard, so the silent rolls shape what you hear. Every pulse of the source pulls the target's next step
  earlier by 5% to 30% of what is left of it (never below 15% of the barre's step after the target's last pulse). The
  pull grows with the chip's pressure, and the wiring does not change. A faster source syncs an even target mostly,
  and a slower one does not. An odd source keeps an even target from settling.

No lane is locked to the bar. A DDR status of 0 grounds the lane, so its rolls and their links are gone. The chip's
own clock is still a plain long-short pulse train and is not one of these rolls; a gong that is not patched to a roll follows it.

## ttrax

`patch.rax_bars` barres (4, 7 or 14, after the real Tetrax, Sidrax and Quatrax) tuned to that many degrees of the
patch's scale, starting at the chip's pitch times arcclk / 800 MHz. Each barre has fixed walls, its own GDDR
channel for the upper one and the next channel for the lower one: `-(0.2 + 0.7 x norm_temp)` and
`0.2 + 0.7 x norm_temp`.

* **Press:** the tone bounces between the fixed walls.
* **Release:** the glitch bounces between the next barre's oscillator and the one after (`glitch_bounds` in
  `music/ttrax.py`). For seven bars this reproduces the Sidrax guide's own Figure 1 (switch 3) exactly; Tetrax
  and Quatrax have no guide here, so the same neighbour pattern is this code's own extension to them. Its
  frequency is the tone's times the fixed window over the current window, between a quarter and eight times.
  Where the upper wall drops below the lower, it rides the lower wall.
* **Chaos:** pressure, plus (in live mode) how fast the chip's temperature and power are moving. The rates add up to
  1 for 1 C/s or 25 W/s, the faster of the two, and the sum stops at 1. Each barre's frequency is swung by the
  oscillator two barres along, by up to 200% at full chaos. Offline there are no rates, so chaos is the pressure.
* **Gestures:** barre b is pressed every four steps of ttrolzer barre b and released half a period later. Each
  envelope peaks on its gesture and falls with a time constant of 0.15 of a period.

## ttquantus

Five triangle oscillators (petals) at ratios 1, 3/2, 5/4, 4/3 and 5/3, each nudged by up to 3% from the seed.
Petal 0 reaches a peak or a valley once per note. Its castle samples a leader petal's triangle at each one and holds the value. The
leader is 2, 3, 4 or 1 places round the ring, picked by the chip's identity. The held value picks a note of the
patch's scale over two octaves. Low prime ratios make the notes repeat, and the nudge makes the pattern drift.
The voice is a sine rounded to 8 bits, held for longer at a lower AXI clock.

## ttcoco

After the Cocoquantus's own delay chips: a pair of crunchy 8-bit echo ICs, each a buffer of a few tens of
kilobits clocked by a VCO, whose length trades directly against its resolution (a shorter delay holds more
bits; a longer one holds fewer, because the same silicon serves both). There are two of them because each
delay line costs a separate piece of hardware.

A Tensix chip has no such limit, and ttcoco is built around that difference: eight parallel taps, one per
GDDR channel, each independently timed and independently crunched, run genuinely in parallel because here a
delay line costs a core's worth of work, not a chip's worth of board space.

* **Source:** `devices.quantussy_tone`, the ttquantussy's plain tone before ttquantus's own 8-bit buffer, patched
  into every tap, as the real instrument's Quantussy patches into its Coco.
* **Per-tap delay:** a hotter GDDR channel echoes faster. 40 ms (hottest) to 900 ms (coolest).
* **Per-tap resolution:** the real chips' own trade, inverted from the delay: the shortest delay holds 8 bits, the
  longest 4.
* **Echo:** each tap is a truncated geometric series of delayed, decaying, independently re-crunched copies: the
  dry signal, then that signal delayed and scaled by the feedback, then delayed twice and scaled by feedback
  squared, and so on until a repeat falls below 1/400 of the first and the series stops. Every repeat is crunched
  to its tap's own bit depth, not to one that keeps getting coarser, which is a simplification of a real feedback
  loop through a lo-fi buffer.
* **Feedback:** pressure, scaled to a ceiling of 0.8 so the echoes always die.
* **Mix:** the eight taps are summed, scaled by 1/sqrt(8), and go through `tanh`.

Not modelled: looping the buffer itself, flip, skip, the Dolby switches, the iron cross and track-and-hold.

## ttavdog

After the AVDog in the Plumbutter guide. The AVDog has a reference oscillator, the dogtone, that drones all the time and is
held silent by a VCA. A trigger pings a slow resonant filter, which rings as an undulation wave: it swings one way, then the other,
and dies away. The undulation's envelope opens the VCA, so the white voice sounds on one half of the swing and the red voice, at
the same pitch with the opposite slope, answers on the other half. The red out is the second trace in the guide's figures.

* **Trigger:** the chip's gong strikes, the same moments that snap the shimmer and ripple the ttdrone.
* **Undulation:** `sin(2 pi rate tau) * exp(-tau / decay)` since the latest strike, with the ttdrone's rate and decay.
* **Dogtone:** at the chip's pitch. A triangle at no pressure, skewed toward an up-sloped saw as pressure rises. The red voice
  uses the inverse slope.
* **Mix:** the two voices are added at a level of 0.5, the red at 0.8 of the white because the guide's mixer has it quieter.

Not modelled: the stereo panning between the two outs, the FM and wave-shape inputs (the purple jacks that modulate the
innards of a module), the Trad/Gonz switch, and the AVDog's audio line input.

## ttchoir

Not from the instrument guides: a human voice, for some humanity in the mix. A voice is a source and a filter. The source is a buzz
at the singer's pitch whose harmonics fall away as `1 / k^1.2` (a glottal pulse). The filter is the vocal tract, five resonances
called formants. Moving the jaw, tongue and lips moves the formants, which is what makes "oo" sound unlike "ee". This is the additive
form of formant synthesis: at each moment the sound is the sum of the source's harmonics, each as loud as the tract lets it be at its
own frequency (`formant_response`).

| Vowel | F1 | F2 | F3 |
|---|---|---|---|
| oo (u) | 300 | 870 | 2240 |
| oh (o) | 570 | 840 | 2410 |
| ah (a) | 730 | 1090 | 2440 |
| eh (e) | 530 | 1840 | 2480 |
| ee (i) | 270 | 2290 | 3010 |

(Hz, a male voice, from Peterson and Barney. F4 and F5 are 3400 and 4400 Hz.)

* **Pitch:** the chip's pitch (C3 to C5 from temperature), moved to the nearest note of the patch's scale so the chips sing in tune with
  each other. A vibrato at 5.5 Hz, 0.25 semitone deep and up to 0.6 under pressure, plus a slow random drift of 0.04 semitone.
* **Vowel:** one syllable is a bar (four beats). Each syllable reads the next GDDR channel, starting from a channel the chip's identity
  picks: a channel hotter than the average channel sings toward "ee", a cooler one toward "oo", and equal temperatures sing "ah". The
  formants glide between vowels over 0.12 s.
* **Voice type:** the chip's identity makes the vocal tract 0%, 8%, 17% or 25% shorter, so every formant is higher (bass toward soprano).
* **Effort:** pressure flattens the glottal slope from `1 / k^1.2` toward `1 / k^0.6`, so a chip near a limit sings brighter.
* **Phrasing:** the voice fades in over 0.1 s and dips to 30% for about 80 ms where one syllable meets the next, as a singer breathes.

Not yet: consonants (fricatives as shaped noise, plosives as a closure and a burst, nasals with an antiresonance), words, breath
noise and a kernel for it. The numbers are textbook values and a first guess at phrasing. None of it has been heard.

## ttsteam and Snare

A 14-step register, one step per Tensix column, stepped at eighth notes. A step fires where its column is
harvested (the mask bit is 0). The live chips each have two harvested columns, so a chip fires two steps in
fourteen and a chip with every column enabled is silent. A hit is white noise rolled off above 4 kHz, with an
envelope that falls to a third in 0.75 s at no pressure and 75 ms at full pressure. Under it, dust pops 15 times
a second for each step of PCIe usage.

The register is in `music/register.py`, apart from the snare, because other devices read it too. Four **steam routes**
patch it into them. A patch lists the routes that are on. The default patch has none, and a drawn patch turns each on with
a 60% chance and always has at least one. Together with ttrollz, the gong and the AVDog they make a rhythm section:

| Route | What it does |
|---|---|
| `strikes` | The register's hits join the ttrollz clock pulses that the gong's trigger response waits on. A strike can then land on a snare hit, and the shimmer's snap, the drone's ripple and the AVDog follow, because they share that clock. The blocking window still spaces the strikes |
| `avdog` | Every hit also pings the AVDog's filter, in addition to the gong strikes, so the white and red voices answer each snare hit |
| `accent` | A ttrollz clock pulse that lands on a register hit (within 5 ms) gets 0.7 of a blip added on top. The clock's long and short steps mean only the pulses on every fourth step can land, so a register that fires on odd steps accents nothing |
| `pitch` | Each step holds one note of the patch's scale, picked from the seed, and the AVDog's dogtone runs at the chip's pitch moved by the note the register holds. The register clocks every step, fired or not, and loops every 14 steps, so it plays a repeating 14-note line |

The routes read only the chip's tempo and Tensix mask on top of what each device already read (`devices.ROUTE_INPUTS`).
The TT-Lang kernels do not follow routes: their host side uses the default patch.

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
| Scale | GDDR spread (hottest minus coolest channel, all chips) | Each 3 C moves one scale along a list of 19: major pentatonic, minor pentatonic, hirajoshi, dorian, whole tone, phrygian dominant, blues, harmonic minor, lydian, mixolydian, phrygian, in sen, iwato, egyptian, double harmonic, hungarian minor, prometheus, augmented, octatonic. Even memory plays the first |
| Gong tuning | Average GDDR temperature divided by average ASIC temperature | Ratio times 10, rounded, picks one of eight: bell, tam-tam, singing bowl, plate, gamelan, glass, cymbal, tubular (hand-picked inharmonic sets, at most five partials) |

**A seed decides the rest:** the lead motif (6 to 12 scale degrees from 0, steps of at most 3, inside a drawn
range), the scale's mode (the same notes started from another degree, so a quiet box that always lands on one
scale still plays a different shape), the gong's beat (0.25 to 2 Hz), a stretch of up to 1.5% on the gong's
partials and a 15% nudge to their levels, the shimmer band (low edge 8 to 15 kHz, 3 to 9 kHz wide), the turn
length (1 to 6 bars), how far non-soloists drop (20% to 50%) and ttrax's bar count. A patch saved before modes
existed has no `mode` and replays as listed.

The seed is a hash of every reading on every chip plus entropy (the clock), so no two runs match. Each run
writes its patch to `patch_<time>.json` beside its log and prints every choice with its reason.
Replay one with `tensixinet --patch FILE`, or fix the seed with `--patch-seed N`. The built-in
`DEFAULT_PATCH` holds the constants from before patches existed (`tensixinet/music/patch.py`).

## Dealing devices to chips

There are always at least ten slots, one per device. Slot `s` goes to chip `s % chips`. One chip
plays all ten. Three chips play ten slots between them. Ten or more chips play one each.

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

## Stereo: a field and not a pair

By default every voice is placed in a stereo field, from the Plumbutter's idea that the AVDogs' white and red voices sit on two channels
and swing between them, and that a patch can send one sound across the field while another stays put. A voice has three coordinates
(`music/space.py`):

* **azimuth**, -1 (hard left) to +1 (hard right), panned with an equal-power law so a voice keeps its loudness as it moves;
* **width**, 0 (a point) to 1: a copy delayed by 11 ms is added to one ear and subtracted from the other, so the two ears differ and a mono
  fold-down loses nothing but the pan law's 3 dB;
* **depth**, 0 (near) to 1 (far): farther is up to 55% quieter, rolled off from 2.5 kHz, and carries a 23 ms early reflection.

**Fixed.** Each chip has a home, spread from -0.75 to +0.75 in the order of the chips (one chip sits in the middle) and nudged by up to 0.04
by its identity. Each device has its own offset from home, width and depth: the lead is near and narrow (+0.15, 0.15, 0.0), the shimmer is wide
and far (0, 1.0, 0.9), the drone wide and behind (0, 1.0, 0.6), the gong far (-0.10, 0.5, 0.7).

**Travelling**, from the chips' own signals:

| What moves | Driven by |
|---|---|
| Every voice orbits its place | the size from the chip's chaos (pressure plus how fast its readings move): 0.04 when calm, up to 0.34; the speed from the clock, one orbit in 20 beats; each chip and device out of step |
| The voice with the floor comes forward and the rest fall back | the turn gains: up to 0.35 deeper at the floor level, in step with the level change |
| ttavdog swings | the undulation: its white voice (the up half) left, its red answer (the down half) right |
| ultrasonic_shimmer pans | at its own beat, 0.2 Hz for each degree of GDDR spread; equal temperatures give a steady tone and no motion |
| gong strikes land | at a new place each strike (the golden ratio apart) and settle to the middle over about a third of a strike interval |
| ttquantus pans | with its pitch: low notes left, high notes right |
| ttrollz hops | from lane to lane, each lane at its own place across the field; the clock's blips are in the middle |
| ttsteam hits | alternate sides |
| ttdrone comes forward | while its ripple swells |

The cables, the turn gains and the device renders act on the mono voices exactly as before; placing comes after, so the same sound is
spread out and not changed. Errors are heard at their chip's place. Each ear is mixed by current on its own. `--mono` is the old sound.
Measured, not heard: 20 s of four mock chips has a left/right correlation of 0.27, the balance moves between -3 and +7 dB over half-second
blocks, and stereo costs about 0.9 s more than mono to render 20 s.

## Evict: when a chip's model unloads

tt-toplike's Defrag view draws a chip's memory as blocks that fill as weights arrive and dissolve (`EVICT`) when the model goes. Its trigger
(`src/animation/defrag.rs`) is a chip that has been running, has *seen a load* (power above 1.5 times its learned rest, or above 30% of its TDP when
no rest has been seen) and has fallen back to about rest (at most 1.2 times, never below 8 W). The "seen a load" gate is what stops an idle chip
from evicting in a loop. `music/evict.py` follows each chip's power the same way, reading by reading (smoothed, with rest learned as a running
minimum from steady readings), and the same moment moves the band:

| Phase | What the band does |
|---|---|
| running | nothing; a box that only ever idles never leaves it |
| **evicting** (8 s) | the chip's voices dissolve right to left across the stereo field: each fades and drifts to the left edge and a little back, starting a step (0.35 s) after the voice to its right, at a drain rate that is a prime multiple (2, 3, 5, 7, 11 ...) as the Defrag view gives each channel. A noise sweep falls in pitch, panning right to left |
| idle | the chip is silent and has the floor about one time in twenty |
| **filling** (8 s) | when load returns the voices come back left to right, a sweep rises, panning left to right, and the chip has 60% of its usual share of the floor until it has settled |

A load that returns during an eviction turns it round from where it had got to. `--mock-load SECONDS` takes each synthetic chip round a load cycle
(14 W rest, 62 W under load, loaded for half the cycle, the chips staggered) so this can be heard without a model; `--no-evict` turns it off. The
log says which chip evicted, the power that decided it, and why. Auditioning with `--only` ignores it. Not heard: only measured.

## Patch cables

The Ciat-Lonbarde guides describe jacks that reach into the innards of a module, so one instrument's signal degrades, mutates
or inflates another's. `music/cables.py` does this between devices. The devices are rendered on their own, so a cable acts on a
finished voice: the source voice drives it, the target voice is changed, and the result goes on to the turn gains and the mix.
A device drives its cables whether or not it has the floor.

| Kind | What it does to the target | Words in the log |
|---|---|---|
| `fm` | reads it through a delay (up to 3 ms) that follows the source waveform | mutates |
| `crunch` | rounds it to steps up to 0.5 wide, wider while the source is loud | degrades |
| `inflate` | multiplies it by up to 3 while the source sounds, with a soft ceiling above 0.9 | inflates |
| `fold` | drives it by up to 5 and reflects what passes 0.9 back inside, which adds overtones | folds |

The patch seed draws five to eight cables between different devices, one per ordered pair, with every kind in use. The choir
always has one cable out and one in, so it bends other voices and is bent by them. The seed is the patch's, so live mode keeps
the same wiring for the whole run and a saved patch replays with it. How hard a cable pushes comes from the chip that plays its
source device: a quarter of the cable's depth from a calm chip, all of it from a chaotic one (`chip_chaos`: pressure plus how
fast its readings move). Load on one chip therefore changes how other devices sound. `--no-cables` turns them off. Not built:
cables that feed back (a target driving its own source), and cables into a device's internals (a chaos knob, a clock) instead of
its output.

## EQ

The devices span about 65 Hz (the gong and drone an octave below the chip's pitch) to 20 kHz (the shimmer). Small
speakers lose the bottom of that range and many people cannot hear the top, so the whole mix goes through one EQ
(`music/eq.py`) after the chips are mixed and before the loudness gain: below 35 Hz is cut, 70 to 250 Hz is lifted
by up to 4 dB, the midrange is left alone, and above 8 kHz the level falls to -14 dB at 14 kHz and -33 dB at 20 kHz.
It is a minimum-phase FIR (4096 taps), so it adds almost no delay and live mode can run it chunk by chunk with the same
result as a single pass. `--no-eq` skips it. The curve is a judgment about ordinary listening and the numbers are in
`EQ_POINTS`; they have not been tuned by ear.

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

* temperature and power get a rate in units per second, over at least 2 s, which feeds ttrax's chaos;
* the memory error and thermal trip counters become what is new, so an error plays once;
* the soloist is drawn from the chips' weights as they are now;
* the turn is rendered with its own seeds and cross-faded for 0.4 s onto the end of the last.

Each turn restarts the devices' clocks, so a gong that rings past the end of its turn fades out instead. The level
follows the music with a running gain toward -18 dBFS RMS. See `DECISIONS.md` for why it is built this way.

