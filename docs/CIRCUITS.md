# What the instrument guides say, and what the code models

The devices are inspired by three Ciat-Lonbarde instruments, using three guides by Josh Singer: the
*Sidrax Guide*, the *Cocoquantus Tasting Guide* and the *Plumbutter & Rolzer Dance Party Guide* (all
revision 1.0). The guides describe circuits that behave differently from ordinary synthesizers, so the
first versions of these devices (written from memory and from the first prototype) were loose
impressions. This page lists what each guide says, what the code now does about it, and what it still
leaves out. The guides are not in the repository. Each module's docstring says which part it follows.

The code models behaviour that can be heard in one chip's telemetry. It does not simulate the circuits, and it is not
a recreation of the instruments. Our devices carry a `tt` prefix for that reason, and this table maps them:

| Ciat-Lonbarde | tensixinet |
|---|---|
| Rolzer (and its rolls) | `ttrollz` (device), `ttrolzer` (module) |
| Sidrax | `ttsidrax` |
| Cocoquantus and its Quantussy | `ttquantus` (device), `ttquantussy` (module) |
| Man With The Red Steam and Snare | `ttsteam` |
| Plumbutter drone, AVDog | `ttdrone` |
| Deerhorn | `tthorn` (not built) |

## Plumbutter and Rolzer

| The guide says | The code does | Where |
|---|---|---|
| The seven Rolzer barres use E6 capacitors of 1 to 10 uF, which gives 192, 122, 84, 56, 39, 27 and 18 BPM. Neighbours run at about 2:3 and every other barre at about 4:9. | Each DDR lane and each group of ethernet links is a roll on its own barre, at that barre's tempo scaled to the chip's tempo. The lanes drift against each other and the bar. | `music/ttrolzer.py` |
| Capacitors are good to 20%, so no two steps last the same. | Every step is jittered by up to 20% from a seed. | `roll_pulse_times` |
| An even roll has a self-reinforcing cycle that swings and stays steady. An odd roll cannot return to its start in one cycle, so it trots and wanders like a double pendulum. | Even rolls alternate long and short steps and hold the tolerance to half. Odd rolls spread their steps unevenly over a cycle and magnify it. | `roll_pulse_times` |
| Grounding a roll's brown node kills the roll. | A DDR status of 0 grounds the lane and silences it. | `mapping.ddr_lane_sides` |
| The Gongue ignores triggers during a delay set by Trigger Response. The delay is a time, so the same setting divides a 100 BPM clock by 2 and a 250 BPM clock by 5. | A gong strike needs a clock pulse that arrives after a blocking window. The chip's identity sets the window, so chips divide their clock differently. | `gate_triggers`, `strike_times` |
| A patch can send one clock to the Ultrasound (the transient), the Gongue (the resonance) and the AVDog (the shock wave) at once (Deconstructed Drum, Shock Wave). | The shimmer snaps and the drone ripples on every gong strike of the same chip. | `ultrasonic_shimmer`, `plumbutter_drone` |
| An AVDog's trigger rings a slow filter at 0.4 to 12 Hz. The swing reverses on every edge, and the white voice sounds on one half and the red voice on the other. | After each strike the drone's level follows a damped swing, rectified because the output is mono. | `shock_wave` |
| The Ultrasound is a switched-capacitor sampler with no anti-alias filter, so ultrasound from the odd rolls folds down into the audible band. | Not modelled. The shimmer stays a pair of tones between 12 and 20 kHz. A 44.1 kHz render cannot hold the signals that would fold down. | |
| Man With The Red Steam is an 8-bit shift register. A clock passes pulses up the stack, and each step can fire a module. The data input, feedback taps and two buttons shape the pattern. | A 14-step register stepped at eighth notes. A step fires where a Tensix column is harvested. Feedback taps and buttons are not modelled. | `music/ttsteam.py` |
| The Snare is quantum noise under an envelope with a decay from 75 ms to 0.75 s, plus dust: a continuous crackle. | The snare is white noise rolled off at 4 kHz. Pressure sets the decay and PCIe usage sets the dust. | `ttsteam` |
| The mixer compresses as channels compete for headroom. | The chorus mixes by current through `tanh`, as before. | `current_mix` |
| The Deerhorn is a two-oscillator theremin played by hand gestures. | Not modelled. Live mode now gives the readings motion, so the rates could drive it. | |

## Sidrax

| The guide says | The code does | Where |
|---|---|---|
| Seven barres, each with an oscillator, tuned by hand as intervals or as a scale. | The seven barres are tuned to the first seven degrees of the patch's scale. | `barre_frequency` |
| An oscillator is a ball bouncing between two bounds. Its frequency is the bounce divided by the bounds. | The frequency is the nominal pitch times the nominal window over the current window, between a quarter and eight times. | `windowed_bounce` |
| When the bounds converge the amplitude falls and the frequency rises. When the upper bound drops below the lower, the oscillator rides the lower bound. | Both happen. | `windowed_bounce` |
| The bounds can be other oscillators. A table (figure 1) says which barres bound which when switch 3, switch 5 or both are on. | The table is in the code. A released barre switches 3 on, so it glitches between two other barres. | `BOUNDS_TABLE`, `sidrax` |
| A press opens one stereo channel and a release the other. In the Pressure Bound patch a press plays the tuned tone and a release switches the bounds on for a decaying glitch. | Each barre plays a tone on its press and a glitch on its release. The output is mono, so the two channels sum. | `gesture_envelopes` |
| The chaos knob cross-modulates the oscillators. At the top it is noise. | Pressure is the chaos knob, and in live mode so is how fast the chip's temperature and power are moving. Each barre's frequency is swung by the barre two along, up to 200% at full chaos. | `windowed_bounce`, `chip_chaos` |
| Barres are played by hand. | Barre b is pressed every four steps of Rolzer barre b, so the seven barres keep different tempos. | `gesture_envelopes` |
| Sequencing the global pitch transposes the whole chord (Chord ChChChChChanges). | The lead's shifts and flips play that role. The ARC clock sets the global pitch. | `lead.py` |
| Stereo, FM inputs, the patch bay, piezo polarity. | Not modelled. | |

## Cocoquantus

| The guide says | The code does | Where |
|---|---|---|
| The Quantussy has five petals, each a triangle oscillator. Each petal's castle samples another petal's triangle at its own peaks and valleys and holds the value. | The first petal's castle picks the note, held for an eighth. | `music/ttquantussy.py` |
| The default leader is two positions counter-clockwise, and two does not divide five, so the chain visits every petal. The castle mod select cycles through leaders 2, 3, 4 and 1. | A chip's identity picks one of those four offsets. | `castle_offset` |
| Petals with low prime ratios make a repeating pattern, and petals that are slightly off make it drift. The castle is not random. | Ratios are 1, 3/2, 5/4, 4/3 and 5/3, each nudged by up to 3% from the seed. The old random sample-and-hold is gone. | `petal_ratios` |
| The delay buffers store 8-bit audio, and the clock is the sample rate, so a slower clock aliases. | The voice is rounded to 8 bits. A slower AXI clock holds each sample longer, from the full rate at 1920 MHz down to a quarter. | `crunch` |
| Delay, looping, feedback, flip, skip, the Dolby switches, the iron cross and the track-and-hold mode. | Not modelled. ttquantus here is the Quantussy's melody through its buffer. | |

## What would come next

* A tthorn, after the Deerhorn, that follows the chips' change over time. Live mode already moves ttsidrax's chaos knob with the rates.
* The Ultrasound's folding needs a sample rate above 96 kHz to be heard as it is in the guide.
* Feedback taps on the shift register would let a chip's PCIe or ethernet traffic set the pattern length.
