# The choir gets consonants: syllables that are not words

Status: draft for review. Written from the conversation of 2026-10-30. Calls the author made without being told are marked **(my call)**.

## Why

`ttchoir` sings vowels only. Its own docstring lists consonants, nasals and breath noise as not modelled. The owner's words: "we're all vowels now. let's get some consonants too." Their favourite module is the ERD/WORM, a Eurorack speech synthesiser (third-party listings say it emulates 1970s to 1990s speech chips, with phonemes and a text-to-speech engine; the maker's own page was not read). It is a source of inspiration, not something to copy. The owner also asked for more range and variance in the choir.

## What it is

The choir sings **syllables that sound like language and are not words**. A syllable is an onset consonant, a vowel, and sometimes a closing consonant. A word can come out only by chance, or when someone plays the instrument with intent. The code holds no word list and no text-to-speech.

## Goals

* **Consonants** in three families **(my call on the inventory)**:
  * fricatives (s, sh, f, h): noise through a narrow moving band;
  * plosives (p, t, k, b, d, g): a short closure of near silence, then a burst, then the vowel's onset glides in;
  * nasals (m, n): the vowel with a spectral notch, as when the mouth closes.
  A soft l or r is a stretch goal. Voiced and unvoiced pairs differ in whether the buzz continues through the closure.
* **Everything is triggered by something.** The owner: "everything should be triggered by _something_ ... patches determine that. if it's on the beat it's because the beat says so. if it's off the beat it's because the beat says so." The choir does not own a clock. Each syllable starts on a trigger from a source the patch names:
  * the gong's strikes, a roll's pulses (`roll_pulses`), the register's hits (`ttsteam`), the looping coco's note onsets (`ttquantus`), the lead's notes, or the ttrax's glitch and barre events;
  * a patch draws which source triggers the choir (and may draw a second one for closing consonants). With the gong or a roll as source the syllables sit on the beat, because the beat says so. With the coco's note onsets or the ttrax they sit where those fall, which is off the beat.
  * A chip's syllable timing is therefore the source's timing. Where a source has a rest, the choir has a rest.
* **Which consonant** is chosen from the chip's readings and the trigger's strength **(my call)**: a hotter GDDR channel leans to sharper sounds (s, t, k), a cooler one to softer ones (m, n, l, b), and a strong trigger leans to plosives while a weak one leans to fricatives and nasals. Same readings and same seed give the same line.
* **Range and variance.** The voice type (the tract's length), the register and the syllable's length drift over a few bars and between chips. The consonant mix changes with them. A voice does not sit on one vowel for a whole bar any more.
* **A speech-chip edge** after the ERD/WORM **(my call)**: pressure adds coarse bit-crushing and sample-rate reduction to the whole voice, so a strained chip sounds more brittle. It is off at rest.
* **A self-oscillation (vocalization) mode, fired by a trigger.** The owner: "triggering a self-oscillation/vocalization mode is a fair feature to add." **(my reading of it)** A trigger from the patch's source can push the vocal tract's resonances into self-oscillation: the formant resonators are driven until they ring on their own, so the voice sings a held, whistling or growling tone with no buzz source, and then decays or is cut. It is the choir's version of feedback and overload, in the spirit of the stage design. The patch draws which source fires it and how rarely (it is a rare event, not the normal voice), and the feedback is bounded by a limiter and a time cap so it stays safe to hear. It is logged when it fires.
* **Playing with intent.** In the Play tab the owner (or anyone) can pick the consonant and vowel directly, so a word can be spelled on purpose. Only the Play tab in the first version **(my call)**.
* **Replayable.** Same patch, same readings, same audio. A saved patch with no choir fields replays exactly as before, because new draws come from their own random stream and the choir with the new fields off is the old choir.
* The device stays inside `ttchoir.py`, with tests first and a mutation check, and every field it reads appears in `DEVICE_INPUTS`.

## Non-goals

* No words, no word list, no text-to-speech, and no language model.
* Not a TT-Lang kernel in the first version.
* No claim that it sounds good until the owner has listened. The first version is measured only.
* Not a change to any other device, and not a change to the stage.

## Anti-goals

* A choir that mumbles real English. A consonant inventory this small can spell a few words by accident, and that is fine, but the patch must not aim at any.
* Consonant noise that lifts the whole mix's level or its high end enough to undo the work on the enya style's dark, close sound. The consonants must sit inside the voice.
* Clicks at the plosive bursts. They need a short envelope.

## Constraints

* The rules in `CLAUDE.md`: tests first, a mutation check, an SPDX header, no `ttnn`, `torch` or `sounddevice` imports, docs that name numbers updated with the code.
* `ttchoir` runs on the CPU and renders per segment in live mode. A new syllable can straddle a segment boundary; the trigger times come from the source's own segment seed (`SEGMENT_SEED_STRIDE`), as the gong and the rolls already share.

## Decisions from the builder check (all my calls, except the one marked open)

* **No trigger, no syllable.** With a trigger source on, the voice is event-gated: it sings a syllable on each trigger and is silent between them, apart from the last vowel dying away over its release. A quiet chip or a source with rests gives a quiet choir, which fits the owner's wish for relief. "The new fields off" means no source: the old choir, on its bar clock, exactly as before.
* **Triggers that land during a syllable** retrigger it: a syllable is cut at the next trigger, its vowel crossfading into the next onset (about 20 ms), and its length is clamped to the gap to the next trigger. The drifting syllable length is the longest it may be. Dense triggers (rolls) therefore give fast, short syllables, and sparse ones (gong) give long ones.
* **Strength.** A source supplies a strength per trigger where it has one (the gong's strike, a note's velocity, a hit's strength). A source with none (the register, the roll pulses) gives a constant 0.5. The builder may add a function in `devices.py` that exposes strengths without changing any device's sound.
* **Three trigger roles, drawn separately by the patch:** the onset source, an optional closing source (it may be the same source; a coda sounds on a second trigger that arrives inside the syllable's gap, and with none there is no coda), and the self-oscillation source. Self-oscillation may fire during a syllable and replaces it. Its time cap is 4 seconds and its peak is limited to -1 dBFS, with the feedback gain cut when the level passes the ceiling.
* **Per chip.** A source is recomputed from the chip's own readings with the same function the source device uses (the gong's `strike_times`, `roll_pulses` and so on). The choir does not follow another chip's device.
* **The bit-crush is inside the choir's own render**, so cables see the crushed voice. It is off at rest.
* **Segment boundaries.** Triggers align across segments through the shared segment seed, and the stitcher's crossfade hides a syllable cut at the boundary. Syllable state does not carry over.
* **`DEVICE_INPUTS`** lists the choir's own fields plus those of every source the patch can draw. The inputs test runs once per source.
* **Decided by the owner (2026-10-30, "not yet. maybe some day"):** no manual "fire" button in the first version. The Play tab picks the next consonant and vowel, and the patch's source still fires them. A fire button, so that a syllable can be spelled on purpose, is a later step and would need a new control type in `knobs.py` and `instrument.py`.

## Left to the builder

* Formant and noise shapes for each consonant (the Klatt cascade and parallel forms are a good reference), burst length, closure length, the notch's frequency and width for the nasals.
* How the choir reads its trigger source's times (a function in `devices.py` beside `strike_times` is the likely home) and how the ttrax's events are exposed, since it has no event list yet.
* The exact mapping from readings and strength to a consonant, as long as it is deterministic and the chip's temperatures move it.
* The names of the new patch fields and styles' values, drawn from their own random stream. The enya style should lean to soft consonants, blues to plosives.
