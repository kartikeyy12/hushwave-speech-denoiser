"""Config validation and derived quantities."""

from __future__ import annotations

import dataclasses
import math

import pytest

from hushwave.config import DEFAULT, Config


def test_default_derived_values() -> None:
    assert DEFAULT.fs == 16_000
    assert DEFAULT.nyquist_hz == 8_000.0
    assert DEFAULT.frame_ms == 32.0
    assert DEFAULT.hop_ms == 16.0
    assert DEFAULT.bin_hz == 31.25
    assert DEFAULT.n_bins == 257
    assert DEFAULT.ms_window_frames == 96
    assert DEFAULT.ms_window_s == pytest.approx(1.536)
    assert DEFAULT.latency_min_samples == 256
    assert DEFAULT.latency_max_samples == 511
    assert DEFAULT.dd_time_constant_ms == pytest.approx(-16.0 / math.log(0.98))
    assert DEFAULT.g_min_db == pytest.approx(-20.0)
    assert DEFAULT.gain_time_constant_ms == pytest.approx(-16.0 / math.log(0.5))


def test_gain_smoother_with_mu_zero_has_no_memory() -> None:
    assert Config(gain_mu=0.0).gain_time_constant_ms == 0.0


def test_config_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT.fs = 8_000  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fs": 0},
        {"pcm_bits": 4},
        {"pcm_bits": 40},
        {"frame_len": 500, "hop": 250},
        {"frame_len": 0, "hop": 0},
        {"hop": 128},
        {"hp_cutoff_hz": 0.0},
        {"hp_cutoff_hz": 8_000.0},
        {"mains_hz": 0.0},
        {"ms_n_sub": 0},
        {"ms_sub_frames": 0},
        {"dd_alpha": 0.0},
        {"dd_alpha": 1.0},
        {"ms_alpha_s": 1.0},
        {"mcra_alpha_p": 0.0},
        {"mcra_delta": 1.0},
        {"noise_floor": 0.0},
        {"ss_a": 0.9},
        {"g_min": 0.0},
        {"g_min": 1.1},
        {"gain_mu": -0.1},
        {"gain_mu": 1.0},
        {"activity_db": 0.0},
    ],
)
def test_invalid_config_is_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        Config(**kwargs)  # type: ignore[arg-type]


def test_other_power_of_two_frames_are_accepted() -> None:
    for frame_len in (256, 512, 1024):
        cfg = Config(frame_len=frame_len, hop=frame_len // 2)
        assert cfg.bin_hz == pytest.approx(16_000 / frame_len)
