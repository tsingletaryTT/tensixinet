# Sound design

## Ciat-Lonbarde ideas used

* **Plumbutter 2:** sources are patched together and mix by current flow.
  Here each chip's share of the mix is its share of the total current, through a saturating
  stage, so piling on chips compresses the sound and never clips it.
* **Rollz:** organic, lopsided pulses. Long and short steps alternate around the beat, and DDR
  status adds polyrhythm lanes.
* **Cocoquantus:** sample-and-hold. Random values are held for an eighth note and quantised to a
  pentatonic scale.
* **Sidrax:** an oscillator that bounces between two bounds. The bounds come from the memory
  temperatures.

## Signal to sound

| Hardware signal | Musical parameter | Range |
|---|---|---|
| `asic_temperature` | Pitch, linear in semitones | 25-100 C maps to C3-C5 |
| `aiclk` | Tempo | 400-1350 MHz maps to 60-180 BPM |
| `power` | Chip volume, exponential curve | 0-200 W maps to 0.3-1.0 |
| `current` | Share of the mix | each chip's fraction of the total |
| GDDR temperatures | Sidrax swing bounds, lead shifts and inversions, shimmer beat rate | every channel the board reports |
| `ddr_status` | Rollz polyrhythm lanes of 3, 4, 5, 7 pulses per bar; status 0 mutes a lane | 4 fields |
| `voltage` | Not used. A test pins that | |

The functions are in `tensixinet/music/mapping.py`. Readings outside a range hold the nearest end.

## The seven devices

| Device | Character | Reads |
|---|---|---|
| Rollz | Lopsided pulse clock plus polyrhythm lanes | temperature, aiclk, DDR status |
| Sidrax | Oscillator bouncing between two bounds | temperature, GDDR temperatures |
| Cocoquantus | Sample-and-hold melody on a pentatonic scale | temperature, aiclk |
| Plumbutter drone | Three-voice C major drone | temperature |
| Lead | Eight-note motif, shifted or inverted bar by bar | temperature, aiclk, GDDR temperatures |
| Gong | Inharmonic strikes, partials 1, 2.76, 5.40, 8.45, each doubled and detuned 0.7 Hz | temperature, aiclk |
| Ultrasonic shimmer | Two close tones in 12-20 kHz | temperature, GDDR temperatures |

## Dealing devices to chips

There are always at least seven slots, one per device. Slot `s` goes to chip `s % chips`. One chip
plays all seven. Three chips play seven slots between them. Seven or more chips play one each.

## Trading solos

A turn lasts two bars at the average tempo. In each turn one slot is featured and the others drop
to 35% volume, with 0.25 s fades. The soloist is drawn in proportion to the power its chip draws,
never repeats back to back, and a quiet chip still gets turns. `music/turns.py` draws from a
golden-ratio sequence, so the same chips always give the same order.

## The lead

Each bar reads the next GDDR channel and compares it with the average channel. Hotter than average
shifts the motif up one scale degree per degree C, at most four. Cooler inverts it around its first
note. Within 1 C it repeats. Even memory temperatures make the lead loop, and uneven ones make it
wander.

## Ultrasonic content

The shimmer sits between 12 and 20 kHz and shrinks to fit below 0.45 of the sample rate. At 44.1 kHz
the top is 19.8 kHz. Use `--sample-rate 96000` for more headroom. Synth partials above Nyquist are
dropped so high voices never alias into the audible band.
