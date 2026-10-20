---
title: tensixinet, what is playing
emoji: 🎛️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Chip-driven synths with a live chart of the mix
---

# tensixinet: what is playing

Tenstorrent chip telemetry played as a band of eleven synthesizers inspired by Ciat-Lonbarde instruments, with a live chart of how the
music is being made: the chips, the devices dealt to them, the patch cables between the devices and the stereo field, with the run log
beside it. Pick a style or a saved patch and press Play. Wear headphones.

**There is no Tenstorrent hardware in this Space.**
- The chips are **synthetic**. Their temperature, power, clock, memory and link readings are made up by the same generator the tests
  use. With a load cycle on, a chip goes from rest to load and back, so you can hear the band change when a chip evicts.
- The gong, shimmer, drone ripple, AVDog, delay mix and mixer can run as the project's real **TT-Lang kernels**, computed by the
  TT-Lang simulator (`tt-lang-sim`) on the CPU. That checks the kernels' arithmetic against the numpy devices. It says nothing about
  the speed or the numerics of a real chip, which differ a little (about 1e-3 at the worst on a Blackhole).
- The real chips, read with `tt-smi`, are an option only on a machine that has them.

Source and the kernels: https://github.com/tsingletaryTT/tensixinet. Inspired by Ciat-Lonbarde and Josh Singer's guides; not a recreation.
