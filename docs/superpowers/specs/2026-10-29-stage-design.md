# The stage: a performance mixer that gives tensixinet quiet and a story

Status: draft for review. Written from the conversation of 2026-10-29. Calls the author made without being told are marked **(my call)**.

## Why

Every device plays all the time, and a turn only decides who is loudest. The result has dynamics between voices and no real rest. The Cocoquantus 2 and the Plumbutter 2 can hold a calm room for a long while, and then a burst of activity means something because of the calm before it. tensixinet should be able to do that: long stretches of quiet, a slow build, a bloom, and a release. The owner's words: "more moments of relief so you can really appreciate when activity is sparking."

## What it is

A **stage** that sits after the devices and cables and before the delay bus, the EQ and the loudness gain. It is a performance mixer in the way the Plumbutter's mixer is one. It is not a twelfth device in the list, because it acts on the other devices' voices. **(my call, following "stage")**

Each voice gets a channel strip: a level, a filter that sweeps (low-pass and high-pass), and sends to a delay, a reverb and a shimmer. A **scene** sets every strip at once. Any of the effects can apply to none, some or all of the voices.

## The story

The stage has its own slow clock and walks through scenes. A scene lasts tens of seconds, and an arc lasts a few minutes. **(my call on the lengths)**

| Scene | What the strips do |
|---|---|
| still | Almost every voice is pulled down and dark. One or two sit in a long reverb. |
| drift | A few voices come and go, wet and filtered. |
| gather | Filters open, delays tighten, more voices come in. |
| bloom | Everything is open and dry. The shimmer is up. |
| release | The voices thin out again through reverb tails. |

The patch draws the order of scenes, how long each lasts and how often the story returns to still. A style limits which scenes it uses and how deep the quiet goes (enya spends most of its time in still and drift, chiptune hardly any). The story is deterministic from the patch seed, so a saved patch replays the same story. A saved patch with no stage field has no stage and replays exactly as before.

Moves between scenes are glides, with each strip moving at its own rate and a little drift, so a change sounds like hands on a mixer and not like a switch. Different voices reach the new scene at slightly different times.

## The clock follows the hardware, through dividers

The stage's clock comes from the chips. The aiclk-to-tempo mapping already gives a tempo, and a chip can report one that is far from where a style wants to live (500 BPM against the 80 enya wants). The stage folds the reading with a divider, a multiplier or a shifter until it lands inside the style's tempo window. Which ratio is used is chosen from a small set (halves, thirds, fourths, doubles and so on), with some hysteresis so that it does not flip when the reading wobbles.

Scene lengths are counted in bars of the folded tempo, so a faster chip moves the story faster and a slower one lets it breathe. Each chip's stage clock has its own shifter (a phase offset from the chip's identity), so the chips do not change scene at the same instant.

Styles set the window: enya gravitates to slow tempos, chiptune to fast ones. The window and the ratio set are in the style, so a new style can say what "slow" means.

## Overload, feedback and using it the wrong way

The stage is allowed to be abused, and the patch decides when. Examples **(my call on the list)**:

* **Overdrive.** A strip's level can go past where it should, into saturation, on a bloom or a spark.
* **Feedback.** Sends can feed back into themselves and into other strips (reverb into delay into a filter at its resonance). Feedback above one is allowed for a limited time and caught by a limiter so that it stays safe to listen to.
* **Misuse.** A still scene can put the filters at self-oscillation on the quiet voices, or send the loud voice only into the reverb. These are patch-drawn, rare, and logged.

The abuse is bounded: a peak limiter after the stage, and a feedback gain that is cut when the level passes a ceiling.

## The chips interrupt

A chip event cuts across the story: a thermal trip, a memory error, a load burst, an eviction or a fill, or a chip's pressure crossing a line. The interruption is a **spark** on that chip's strips. They snap open and dry for a few seconds, then glide back to whatever scene the story is in. The other chips keep following the story. **(my call)**

A spark is a moment inside the arc. A load that lasts (the chip's pressure stays high for a stretch, which the evict tracker already measures) can also move the story forward one scene, so a long burst pushes drift into gather. A single event does not. **(my call; the owner said "the chips can interrupt it" and did not say whether the story moves)**

## Interfaces and flags

* `--no-stage` turns it off. The default patch and old saved patches have no stage.
* `Patch.stage` (0 to 1) is how much of the story and quiet is used. Drawn from its own random stream, so no existing draw moves.
* The log has a stage line (scene, tempo ratio, sparks).
* The Play tab's knobs and the app's chart can show the scene later. Not in the first version.

## Goals

* A measurable amount of quiet: with the stage on and the enya style, a good share of a run is more than 12 dB below its loud parts, where it is now close to none (measure the loudness curve of several runs first and use the result as the baseline).
* Natural movement: no audible clicks at scene changes, and glides of seconds.
* Replayable: the same patch and readings give the same story.
* Works in offline and live mode with the same code.

## Non-goals

* Not a TT-Lang kernel. The stage runs on the CPU.
* Not a MIDI source in the first version.
* No claim that it sounds good until the owner has listened. The first version is measured only.
* Not a change to any device's own sound.

## Anti-goals

* A mixer that makes every style quiet, or one that ducks the music for good. The loud scenes must still be loud.
* Feedback that clips or damages speakers. The limiter is not optional.

## Left to the builder

* The shapes of the filters and effects (reuse `cocobus` reverb and taps and the EQ's filters where they fit).
* The exact ratio set, the hysteresis, the scene lengths and the glide times, to be tuned by the measured loudness curve.
* How the stage keeps its place across live turns.
* How the tests prove the story is replayable and that a spark does not move the other chips.
