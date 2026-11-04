from __future__ import annotations

import numpy as np
import pytest

from hushwave.config import DEFAULT, Config
from hushwave.enhancer import enhance
from hushwave.synth import mix_at_snr
from hushwave.systems import (
    System,
    b5_from,
    config_for_frame_ms,
    make_ablations,
    make_baselines,
    resolve,
    run_system,
    system_from_dict,
    system_to_dict,
)

FS = 16_000


def _mixture(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.arange(2 * FS) / FS
    clean = 0.2 * np.sin(2 * np.pi * 220 * t) * (1 + np.sin(2 * np.pi * 3 * t))
    mix = mix_at_snr(clean, rng.standard_normal(clean.size), 5.0)
    return clean, mix.noise, mix.noisy


def _b5() -> System:
    base = System("x", "x", "gain", rule="lsa", estimator="mcra", prefilter="spectral")
    return b5_from(base.with_gain(g_min=0.1, gain_mu=0.5).with_est(rise_db_per_s=3.0))


def test_frame_scaling_keeps_durations() -> None:
    reference = config_for_frame_ms(32)
    assert reference.frame_len == 512 and reference.hop == 256
    for ms in (16, 64):
        cfg = config_for_frame_ms(ms)
        assert cfg.frame_len == ms * 16
        assert cfg.ms_window_s == pytest.approx(DEFAULT.ms_window_s, rel=0.05)
        assert cfg.dd_time_constant_ms == pytest.approx(DEFAULT.dd_time_constant_ms, rel=1e-9)
        assert cfg.gain_time_constant_ms == pytest.approx(DEFAULT.gain_time_constant_ms, rel=1e-9)
    with pytest.raises(ValueError):
        config_for_frame_ms(32, Config(fs=11025, frame_len=512, hop=256))


def test_system_validation_and_serialisation() -> None:
    with pytest.raises(ValueError):
        System("a", "a", "magic")
    with pytest.raises(ValueError):
        System("a", "a", "gain", prefilter="sharp")
    system = _b5().with_gain(dd_alpha=0.9).with_est(delta=4.0)
    assert system.gain == (("dd_alpha", 0.9), ("g_min", 0.1), ("gain_mu", 0.5))
    assert system_from_dict(system_to_dict(system)) == system
    restored = system_from_dict({**system_to_dict(system), "gain": [["g_min", 0.2]]})
    assert restored.gain == (("g_min", 0.2),)


def test_resolve_scales_rates_and_enables_the_spectral_mask() -> None:
    system = _b5().with_gain(dd_alpha=0.98).with_est(alpha_s=0.9, sub_frames=16.0)
    cfg32, gain32, est32 = resolve(system)
    assert gain32["spectral_hp"] == 1.0 and gain32["dd_alpha"] == pytest.approx(0.98)
    cfg16, gain16, est16 = resolve(System(**{**system.__dict__, "frame_ms": 16}))
    assert cfg16.hop == 128
    assert gain16["dd_alpha"] == pytest.approx(0.98**0.5)
    assert est16["alpha_s"] == pytest.approx(0.9**0.5)
    assert est16["sub_frames"] == 32.0
    assert est32["sub_frames"] == 16.0
    assert cfg32.hop == 256


def test_unprocessed_and_gain_systems(rng: np.random.Generator) -> None:
    clean, noise, noisy = _mixture(rng)
    out = run_system(System("B0", "raw", "unprocessed"), clean, noise, noisy)
    np.testing.assert_array_equal(out.enhanced, noisy)
    assert out.speech_component is None
    b5 = _b5()
    result = run_system(b5, clean, noise, noisy)
    assert result.speech_component is not None
    direct = enhance(
        noisy,
        rule="lsa",
        estimator="mcra",
        prefilter=False,
        gain_overrides={"g_min": 0.1, "gain_mu": 0.5, "spectral_hp": 1.0},
        estimator_overrides={"rise_db_per_s": 3.0},
    )
    np.testing.assert_allclose(result.enhanced, direct, atol=1e-9)
    biquad = run_system(System("b", "b", "gain", prefilter="biquad"), clean, noise, noisy)
    assert not np.allclose(
        biquad.enhanced,
        run_system(System("n", "n", "gain", prefilter="none"), clean, noise, noisy).enhanced,
    )
    short = run_system(System("f", "f", "gain", frame_ms=16), clean, noise, noisy)
    long = run_system(System("f", "f", "gain", frame_ms=64), clean, noise, noisy)
    for item in (short, long):
        assert item.enhanced.shape == noisy.shape and np.all(np.isfinite(item.enhanced))


def test_library_baselines_run(rng: np.random.Generator) -> None:
    clean, noise, noisy = _mixture(rng)
    wiener = run_system(System("B2", "w", "scipy_wiener", window=9), clean, noise, noisy)
    assert wiener.enhanced.shape == noisy.shape and wiener.speech_component is None
    pytest.importorskip("noisereduce")
    reduced = run_system(System("B3", "n", "noisereduce"), clean, noise, noisy)
    assert reduced.enhanced.shape == noisy.shape


def test_baselines_share_the_pipeline_of_b5() -> None:
    b5 = _b5()
    systems = make_baselines(b5, (("ss_a", 3.0), ("gain_mu", 0.0)), (("g_min", 0.2),), 31)
    assert [s.name for s in systems] == ["B0", "B1", "B2", "B3", "B4", "B5"]
    by_name = {s.name: s for s in systems}
    for name in ("B1", "B4"):
        shared = by_name[name]
        assert (shared.estimator, shared.prefilter, shared.est, shared.frame_ms) == (
            b5.estimator,
            b5.prefilter,
            b5.est,
            b5.frame_ms,
        )
    assert by_name["B1"].rule == "subtraction" and by_name["B4"].rule == "ml_wiener"
    assert by_name["B2"].window == 31
    assert all(s.external for s in systems)


def test_ablations_change_one_thing_and_are_unique() -> None:
    b5 = _b5()
    ablations = make_ablations(b5)
    names = [a.name for a in ablations]
    assert len(set(names)) == len(names)
    assert all(n.startswith("A:") for n in names)
    assert not any(a.external for a in ablations)
    structures = [a.structure() for a in ablations]
    assert len(set(structures)) == len(structures) and b5.structure() not in structures
    assert {"A:frame-16ms", "A:frame-64ms", "A:rise-off", "A:est-minstats"} <= set(names)
    assert "A:prefilter-spectral" not in names
    assert {"A:prefilter-none", "A:prefilter-biquad"} <= set(names)
    plain = b5_from(System("x", "x", "gain", estimator="minstats"))
    assert "A:rise-3" in {a.name for a in make_ablations(plain)}
    assert "A:est-mcra" in {a.name for a in make_ablations(plain)}
