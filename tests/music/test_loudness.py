# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""One gain for a whole render, so every run lands at the same level and keeps its own dynamics."""
import numpy as np
import pytest

from tensixinet.music.loudness import PEAK_LIMIT, TARGET_RMS, loudness_gain, normalise


def _tone(amp, n=8000):
    return amp * np.sin(2 * np.pi * 220 * np.arange(n) / 8000)


def _rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def test_the_target_is_minus_eighteen_dbfs_rms_and_the_peak_limit_minus_one_dbfs():
    assert TARGET_RMS == pytest.approx(10 ** (-18 / 20), rel=0.01)
    assert PEAK_LIMIT == pytest.approx(10 ** (-1 / 20), rel=0.01)


def test_a_quiet_render_is_brought_up_to_the_target_level():
    out, gain = normalise(_tone(0.05))
    assert gain > 1 and _rms(out) == pytest.approx(TARGET_RMS, rel=1e-3)


def test_a_loud_render_is_brought_down_to_the_same_level():
    out, gain = normalise(_tone(0.9) * 1.0)
    assert gain < 1 and _rms(out) == pytest.approx(TARGET_RMS, rel=1e-3)


def test_two_renders_of_different_loudness_end_at_the_same_level():
    assert _rms(normalise(_tone(0.04))[0]) == pytest.approx(_rms(normalise(_tone(0.4))[0]), rel=1e-3)


def test_a_peaky_render_is_held_to_the_peak_limit_before_it_reaches_the_target():
    x = np.zeros(8000)
    x[::800] = 0.5  # sparse clicks: low rms for their peak
    out, gain = normalise(x)
    assert np.abs(out).max() == pytest.approx(PEAK_LIMIT, rel=1e-6) and _rms(out) < TARGET_RMS


def test_the_gain_is_capped_so_a_near_silent_render_is_not_amplified_into_noise():
    assert loudness_gain(_tone(1e-6)) == pytest.approx(10.0)  # +20 dB


def test_silence_is_left_alone():
    out, gain = normalise(np.zeros(100))
    assert gain == 1.0 and not np.any(out)
    assert loudness_gain(np.zeros(0)) == 1.0


def test_one_gain_for_the_whole_render_keeps_the_dynamics():
    x = np.concatenate([_tone(0.02), _tone(0.2)])  # a quiet half and a loud half, ten to one
    out, _ = normalise(x)
    assert _rms(out[8000:]) / _rms(out[:8000]) == pytest.approx(10.0, rel=1e-3)


def test_the_result_has_the_input_shape_and_is_float32():
    out, _ = normalise(_tone(0.1).astype(np.float32))
    assert out.shape == (8000,) and out.dtype == np.float32


# --- live: a gain that follows the music as it plays --------------------------------------------------------

from tensixinet.music.loudness import SMOOTHING, LiveGain  # noqa: E402


def test_the_first_chunk_gets_the_gain_a_whole_render_would_get():
    out, gain = LiveGain().apply(_tone(0.05))
    assert gain == pytest.approx(loudness_gain(_tone(0.05)))
    assert _rms(out) == pytest.approx(TARGET_RMS, rel=0.02)  # a first chunk has nothing to ramp from, so its gain is flat


def test_later_gains_move_part_of_the_way_to_each_chunks_own_gain_in_the_log_domain():
    live = LiveGain()
    _, first = live.apply(_tone(0.05))
    _, second = live.apply(_tone(0.2))  # four times louder, so its own gain is a quarter of the first
    target = loudness_gain(_tone(0.2))
    assert second == pytest.approx(first * (target / first) ** SMOOTHING)
    assert target < second < first


def test_the_gain_ramps_across_a_chunk_so_a_change_has_no_step():
    live = LiveGain()
    _, first = live.apply(np.full(1000, 0.05))
    out, second = live.apply(np.full(1000, 0.2))
    assert out[0] == pytest.approx(0.2 * first, rel=1e-3) and out[-1] == pytest.approx(0.2 * second, rel=1e-2)
    assert np.all(np.diff(out) <= 1e-9) or np.all(np.diff(out) >= -1e-9)  # one direction, no overshoot


def test_the_peak_limit_holds_even_when_the_gain_was_high_a_moment_ago():
    live = LiveGain()
    live.apply(_tone(0.02))  # a quiet start sets a high gain
    out, gain = live.apply(_tone(0.95))  # then a sudden loud chunk
    assert float(np.max(np.abs(out))) <= PEAK_LIMIT + 1e-9
    assert gain <= PEAK_LIMIT / 0.95 + 1e-9  # the gain itself comes down at once, not only the clipped samples


def test_silence_leaves_the_gain_where_it_was():
    live = LiveGain()
    _, before = live.apply(_tone(0.1))
    out, after = live.apply(np.zeros(500))
    assert after == before and not out.any()


def test_a_run_that_starts_in_silence_gets_its_gain_from_its_first_sound():
    live = LiveGain()
    live.apply(np.zeros(500))
    _, gain = live.apply(_tone(0.05))
    assert gain == pytest.approx(loudness_gain(_tone(0.05)))
