"""Gain rules: ranges, monotonicity, limits, decision-directed recursion, smoothing."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import exp1

from hushwave.config import DEFAULT, Config
from hushwave.gains import (
    RULES,
    GainStage,
    compute_gains,
    lsa_gain,
    lsa_gain_raw,
    noise_only_floor_escape,
    subtraction_gain,
    wiener_gain,
)

positive = st.floats(min_value=1e-6, max_value=1e6, allow_nan=False)


# ---------------------------------------------------------------- pure rules
def test_subtraction_matches_formula() -> None:
    gamma = np.array([0.5, 1.0, 4.0, 9.0, 100.0])
    expected = np.clip(1.0 - 2.0 / np.sqrt(gamma), 0.1, 1.0)
    np.testing.assert_allclose(subtraction_gain(gamma, 2.0, 0.1), expected)


def test_subtraction_zero_gamma_is_floor() -> None:
    assert subtraction_gain(0.0, 2.0, 0.1) == pytest.approx(0.1)


def test_subtraction_is_nondecreasing_in_gamma() -> None:
    gamma = np.logspace(-3, 4, 400)
    assert np.all(np.diff(subtraction_gain(gamma, 2.0, 0.1)) >= 0.0)


def test_wiener_matches_formula_and_limits() -> None:
    xi = np.array([0.0, 1.0, 3.0, 99.0])
    np.testing.assert_allclose(wiener_gain(xi, 1e-3), [1e-3, 0.5, 0.75, 0.99])
    assert wiener_gain(np.inf, 0.1) == pytest.approx(1.0)
    assert wiener_gain(1e12, 0.1) == pytest.approx(1.0)


def test_wiener_is_strictly_increasing_above_the_floor() -> None:
    xi = np.logspace(-2, 4, 300)
    assert np.all(np.diff(wiener_gain(xi, 1e-6)) > 0.0)


def test_lsa_raw_matches_closed_form() -> None:
    xi, gamma = 2.0, 3.0
    g2 = xi / (1 + xi)
    expected = g2 * math.exp(0.5 * float(exp1(g2 * gamma)))
    assert lsa_gain_raw(xi, gamma) == pytest.approx(expected)


def test_lsa_raw_example_from_theory_exceeds_one_and_is_clipped() -> None:
    assert lsa_gain_raw(1.0, 0.01) == pytest.approx(5.3116, abs=1e-3)
    assert lsa_gain(1.0, 0.01, 0.1) == 1.0


def test_lsa_raw_is_at_least_wiener() -> None:
    xi, gamma = np.meshgrid(np.logspace(-3, 4, 60), np.logspace(-3, 3, 60))
    assert np.all(lsa_gain_raw(xi, gamma) >= xi / (1.0 + xi) - 1e-15)


def test_lsa_raw_increases_with_xi_and_decreases_with_gamma() -> None:
    xi, gamma = np.meshgrid(np.logspace(-3, 4, 60), np.logspace(-3, 3, 60))
    raw = lsa_gain_raw(xi, gamma)
    assert np.all(np.diff(raw, axis=1) > 0.0)
    assert np.all(np.diff(raw, axis=0) <= 1e-12)


def test_lsa_tends_to_wiener_for_large_gamma_and_to_one_for_large_xi() -> None:
    assert lsa_gain_raw(1.0, 1e6) == pytest.approx(0.5, rel=1e-6)
    assert lsa_gain(1e9, 1.0, 0.1) == pytest.approx(1.0, abs=1e-6)


def test_lsa_edge_values_are_never_nan() -> None:
    assert lsa_gain_raw(0.0, 5.0) == 0.0
    assert lsa_gain_raw(0.0, 0.0) == 0.0
    assert lsa_gain_raw(1.0, 0.0) == np.inf
    assert lsa_gain(1.0, 0.0, 0.1) == 1.0
    assert np.all(
        np.isfinite(lsa_gain(np.array([0.0, 1.0, np.inf]), np.array([0.0, 0.0, 0.0]), 0.1))
    )


@given(xi=positive, gamma=positive, g_min=st.floats(min_value=1e-3, max_value=1.0))
@settings(max_examples=200, deadline=None)
def test_every_rule_stays_in_range(xi: float, gamma: float, g_min: float) -> None:
    for gain in (
        subtraction_gain(gamma, 2.0, g_min),
        wiener_gain(xi, g_min),
        lsa_gain(xi, gamma, g_min),
    ):
        assert np.isfinite(gain)
        assert g_min - 1e-12 <= float(gain) <= 1.0


def test_lsa_is_at_least_wiener_after_clipping() -> None:
    xi, gamma = np.meshgrid(np.logspace(-3, 4, 40), np.logspace(-3, 3, 40))
    assert np.all(lsa_gain(xi, gamma, 0.1) >= wiener_gain(xi, 0.1) - 1e-15)


@pytest.mark.parametrize(
    "call",
    [
        lambda: subtraction_gain(-1.0, 2.0, 0.1),
        lambda: subtraction_gain(1.0, 0.5, 0.1),
        lambda: subtraction_gain(1.0, 2.0, 0.0),
        lambda: subtraction_gain(np.nan, 2.0, 0.1),
        lambda: wiener_gain(-0.1, 0.1),
        lambda: wiener_gain(1.0, 1.5),
        lambda: lsa_gain_raw(-1.0, 1.0),
        lambda: lsa_gain_raw(1.0, -1.0),
        lambda: lsa_gain(1.0, 1.0, 0.0),
        lambda: noise_only_floor_escape(0.5, 0.1),
        lambda: noise_only_floor_escape(2.0, 1.0),
    ],
)
def test_pure_rules_reject_bad_arguments(call: object) -> None:
    with pytest.raises(ValueError):
        call()  # type: ignore[operator]


def test_noise_only_floor_escape_matches_monte_carlo(rng: np.random.Generator) -> None:
    gamma = rng.exponential(1.0, 2_000_000)
    for a, g_min in ((1.0, 0.1), (2.0, 0.1), (2.0, 0.3)):
        measured = float(np.mean(subtraction_gain(gamma, a, g_min) > g_min + 1e-12))
        assert measured == pytest.approx(noise_only_floor_escape(a, g_min), abs=2e-3, rel=0.05)


# ---------------------------------------------------------------- GainStage
def _noise_power(rng: np.random.Generator, frames: int, bins: int = 33) -> np.ndarray:
    return rng.exponential(1.0, (frames, bins))


@pytest.mark.parametrize("rule", RULES)
def test_stage_output_is_finite_and_in_range(rule: str, rng: np.random.Generator) -> None:
    bins = 33
    power = np.concatenate(
        [_noise_power(rng, 100, bins), 1e4 * _noise_power(rng, 50, bins), np.zeros((5, bins))]
    )
    gains = compute_gains(power, np.ones_like(power), rule)
    assert np.all(np.isfinite(gains))
    assert gains.min() >= DEFAULT.g_min - 1e-12
    assert gains.max() <= 1.0 + 1e-12


def test_g1_depends_only_on_current_gamma(rng: np.random.Generator) -> None:
    stage_a = GainStage("subtraction", 33, gain_mu=0.0)
    stage_b = GainStage("subtraction", 33, gain_mu=0.0)
    ones = np.ones(33)
    for _ in range(20):
        stage_a.process(100.0 * rng.exponential(1.0, 33), ones)
        stage_b.process(rng.exponential(1.0, 33), ones)
    probe = rng.exponential(5.0, 33)
    np.testing.assert_array_equal(stage_a.process(probe, ones), stage_b.process(probe, ones))


def test_ml_wiener_equals_power_subtraction(rng: np.random.Generator) -> None:
    gamma = rng.exponential(2.0, (50, 33))
    gains = compute_gains(gamma, np.ones_like(gamma), "ml_wiener", gain_mu=0.0, g_min=1e-3)
    np.testing.assert_allclose(gains, np.maximum(1.0 - 1.0 / np.maximum(gamma, 1.0), 1e-3))


def test_decision_directed_first_frame_uses_only_the_instantaneous_term() -> None:
    stage = GainStage("wiener", 3, gain_mu=0.0, g_min=1e-3)
    stage.process(np.array([1.0, 5.0, 101.0]), np.ones(3))
    np.testing.assert_allclose(stage.xi, (1.0 - DEFAULT.dd_alpha) * np.array([0.0, 4.0, 100.0]))


def test_decision_directed_recursion_matches_hand_calculation() -> None:
    stage = GainStage("wiener", 2, gain_mu=0.0, g_min=1e-6)
    ones = np.ones(2)
    gamma1 = np.array([11.0, 3.0])
    g1 = stage.process(gamma1, ones)
    xi1 = (1 - 0.98) * (gamma1 - 1)
    np.testing.assert_allclose(g1, xi1 / (1 + xi1))
    gamma2 = np.array([4.0, 2.0])
    stage.process(gamma2, ones)
    np.testing.assert_allclose(stage.xi, 0.98 * g1**2 * gamma1 + 0.02 * (gamma2 - 1))


def test_decision_directed_suppresses_noise_only_cells_better_than_ml(
    rng: np.random.Generator,
) -> None:
    power = _noise_power(rng, 400, 129)
    ones = np.ones_like(power)
    floor = DEFAULT.g_min
    escaped = {}
    for rule in ("ml_wiener", "wiener"):
        gains = compute_gains(power, ones, rule, gain_mu=0.0)[100:]
        escaped[rule] = float(np.mean(gains > 2.0 * floor))
    assert escaped["ml_wiener"] > 0.2
    assert escaped["wiener"] < 0.01


def test_lsa_lets_more_noise_through_than_wiener_by_construction(rng: np.random.Generator) -> None:
    power = _noise_power(rng, 400, 129)
    ones = np.ones_like(power)
    wiener_mean = compute_gains(power, ones, "wiener", gain_mu=0.0)[100:].mean()
    lsa_mean = compute_gains(power, ones, "lsa", gain_mu=0.0)[100:].mean()
    assert lsa_mean > wiener_mean


def test_decision_directed_tracks_a_strong_stationary_signal() -> None:
    stage = GainStage("wiener", 4, gain_mu=0.0)
    gamma = np.full(4, 101.0)
    for _ in range(12):
        stage.process(gamma, np.ones(4))
    assert stage.xi[0] == pytest.approx(100.0, rel=0.05)


def test_temporal_smoothing_follows_the_exact_recursion() -> None:
    mu = 0.5
    stage = GainStage("subtraction", 2, gain_mu=mu, g_min=0.1)
    ones = np.ones(2)
    quiet = np.zeros(2)  # gamma = 0 -> floor
    loud = np.full(2, 1e6)  # gamma = 1e6 -> gain about 0.998
    first = stage.process(quiet, ones)
    np.testing.assert_allclose(first, 0.1)
    target = float(subtraction_gain(1e6, DEFAULT.ss_a, 0.1))
    for n in range(1, 6):
        out = stage.process(loud, ones)
        np.testing.assert_allclose(out, target - (target - 0.1) * mu**n)


def test_smoothing_reduces_a_one_frame_spike_by_one_minus_mu() -> None:
    stage = GainStage("subtraction", 1, gain_mu=0.5, g_min=0.1)
    one = np.ones(1)
    stage.process(np.zeros(1), one)
    spike = stage.process(np.full(1, 1e6), one)
    full = float(subtraction_gain(1e6, DEFAULT.ss_a, 0.1))
    assert spike[0] - 0.1 == pytest.approx(0.5 * (full - 0.1))


def test_mu_zero_is_identity() -> None:
    stage = GainStage("subtraction", 2, gain_mu=0.0)
    out = stage.process(np.array([0.0, 1e6]), np.ones(2))
    np.testing.assert_allclose(out, subtraction_gain(np.array([0.0, 1e6]), DEFAULT.ss_a, 0.1))


def test_reset_restores_the_initial_state(rng: np.random.Generator) -> None:
    power = _noise_power(rng, 30, 17)
    stage = GainStage("lsa", 17)
    first = np.stack([stage.process(p, np.ones(17)) for p in power])
    stage.reset()
    second = np.stack([stage.process(p, np.ones(17)) for p in power])
    np.testing.assert_array_equal(first, second)


def test_stage_returns_a_copy() -> None:
    stage = GainStage("subtraction", 3, gain_mu=0.5)
    out = stage.process(np.ones(3), np.ones(3))
    out[:] = 99.0
    assert np.all(stage.process(np.ones(3), np.ones(3)) < 2.0)


def test_floor_of_one_gives_unit_gain(rng: np.random.Generator) -> None:
    power = _noise_power(rng, 50, 17)
    for rule in RULES:
        gains = compute_gains(power, np.ones_like(power), rule, g_min=1.0)
        np.testing.assert_array_equal(gains, 1.0)


@pytest.mark.parametrize(
    "build",
    [
        lambda: GainStage("nonsense", 8),
        lambda: GainStage("lsa", 0),
        lambda: GainStage("lsa", 8, nonsense=1.0),
        lambda: GainStage("lsa", 8, ss_a=0.5),
        lambda: GainStage("lsa", 8, g_min=0.0),
        lambda: GainStage("lsa", 8, gain_mu=1.0),
        lambda: GainStage("lsa", 8, dd_alpha=1.0),
    ],
)
def test_stage_rejects_bad_construction(build: object) -> None:
    with pytest.raises(ValueError):
        build()  # type: ignore[operator]


def test_stage_rejects_bad_frames() -> None:
    stage = GainStage("lsa", 4)
    good = np.ones(4)
    for power, noise in (
        (np.ones(3), good),
        (good, np.ones(5)),
        (np.array([1.0, np.nan, 1.0, 1.0]), good),
        (np.array([1.0, -1.0, 1.0, 1.0]), good),
        (good, np.array([1.0, 0.0, 1.0, 1.0])),
        (good, np.array([1.0, np.inf, 1.0, 1.0])),
    ):
        with pytest.raises(ValueError):
            stage.process(power, noise)


def test_compute_gains_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="matching"):
        compute_gains(np.ones((3, 4)), np.ones((3, 5)), "lsa")
    with pytest.raises(ValueError, match="no frames"):
        compute_gains(np.ones((0, 4)), np.ones((0, 4)), "lsa")


def test_config_overrides_are_used() -> None:
    cfg = Config(ss_a=3.0, g_min=0.2, gain_mu=0.0)
    stage = GainStage("subtraction", 2, cfg)
    assert (stage.a, stage.g_min, stage.mu) == (3.0, 0.2, 0.0)
    out = stage.process(np.zeros(2), np.ones(2))
    np.testing.assert_allclose(out, 0.2)
