# Contributing

This applies to people and to coding agents. The rule that matters most: write the test first, watch
it fail, then write the code, then break the code on purpose and watch a test go red.

## Setup and checks

```bash
pip install -e .[dev]
python3 -m pytest        # about 1 s, no hardware
```

Run the suite in a bare virtualenv as well (`python3 -m venv v && v/bin/pip install .[dev]`), so
nothing on your machine hides a missing dependency.

## The loop

1. **Name the change that would make the test fail**, then write one test for one behaviour.
2. **Run it and confirm it fails for the right reason** (the feature is missing; an import error or a typo is the wrong failure).
3. **Write the least code that passes.** Run the whole suite, since one passing file says little.
4. **Mutate.** Swap a `min` for a `max`, drop a term, remove a call, change a constant, and run the
   suite. If it stays green, a test is missing or too weak. Add one, then mutate again. Make sure the
   suite was green before you mutated: a test that is already red catches everything.
5. **Check that scripted edits changed something.** A search-and-replace that matches nothing reports
   success.
6. **Update the docs that name numbers** in the same change.

## Add a device

1. Write the device as `f(telemetry, duration, sample_rate, seed) -> float32 array`, bounded by 1 and
   deterministic for a seed. Put it in `tensixinet/music/devices.py`, or in its own module under
   `music/` if it is large (like `lead.py`).
2. Add it to `DEVICE_INPUTS` in `devices.py` with exactly the telemetry fields it reads.
   `tests/narrate/test_explain.py` fails if that list and the audio disagree.
3. Add it to `DEVICES` in `music/chorus.py`. Slots, solos and the log pick it up from there.
4. Write its tests in `tests/music/`. Test the property that makes it that device (inharmonic
   partials, beating, a shift that raises pitch). A test that only checks for an array would pass
   for any device.
5. If a field should change what the log says, extend `narrate/explain.py` and test the text.
6. Update `docs/SOUND.md` and the README table.

## Change a mapping

Edit `music/mapping.py` and its tests together, then `docs/SOUND.md`. The README and SOUND tables must
match the code.

## Hardware rules

* Reading telemetry (`tt-smi -s`, `tt-toplike --backend hybrid`) and HTTP to the model endpoint do
  not open a device.
* Anything that opens `/dev/tenstorrent/*` needs a lease on a shared machine
  (`gozer run --chips N --who ... --reason ... -- command`). Importing `ttnn`, `ttl` or `ttml` opens
  the device, so nothing in `tensixinet/` or `tests/` may import them. `tests/test_hygiene.py`
  enforces that.
* A test that needs hardware gets `@pytest.mark.hardware` and runs under a lease.

## Style

Plain English in docs, comments and commit messages. Short sentences. Say what a number does.
Comments explain why. Keep the terminal output to a left bar with no right border.
