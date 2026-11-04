"""The systems that are compared: B0 to B5 and the ablation variants (THEORY section 8.3).

A system is a small frozen description; ``run_system`` turns it into an enhanced signal. Gain-based
systems (B1, B4, B5, ablations) run through ``enhancer.gain_components``, which gives the output
and the speech component for the speech-distortion metric from one pass.

Frame-length ablation. Changing the frame length changes the hop, so every quantity that is
defined per frame would silently change its duration. ``config_for_frame_ms`` keeps durations
constant: the minimum window stays about 1.5 s long (sub-window length scales with 1 / hop), and
every smoothing constant a is replaced by a^(hop / 256), which keeps its time constant in seconds.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hushwave.baselines import b2_scipy_wiener, b3_noisereduce
from hushwave.config import DEFAULT, Config
from hushwave.enhancer import enhance, gain_components

FloatArray = NDArray[np.float64]
PREFILTERS = ("none", "biquad", "spectral")
KINDS = ("unprocessed", "gain", "scipy_wiener", "noisereduce")
REFERENCE_HOP = DEFAULT.hop
_RATE_KEYS_GAIN = ("dd_alpha", "gain_mu")
_RATE_KEYS_EST = ("alpha_s", "alpha_p", "alpha_d")

Pairs = tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class System:
    name: str
    label: str
    kind: str
    rule: str = "lsa"
    estimator: str = "mcra"
    prefilter: str = "none"
    gain: Pairs = ()
    est: Pairs = ()
    frame_ms: int = 32
    window: int = 15
    external: bool = False

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown system kind {self.kind!r}; choose from {KINDS}")
        if self.prefilter not in PREFILTERS:
            raise ValueError(f"unknown prefilter {self.prefilter!r}; choose from {PREFILTERS}")

    def with_gain(self, **values: float) -> System:
        merged = dict(self.gain)
        merged.update(values)
        return replace(self, gain=tuple(sorted(merged.items())))

    def with_est(self, **values: float) -> System:
        merged = dict(self.est)
        merged.update(values)
        return replace(self, est=tuple(sorted(merged.items())))

    def structure(self) -> tuple[Any, ...]:
        """Everything that defines behaviour (names and flags excluded), for de-duplication."""
        return (
            self.kind,
            self.rule,
            self.estimator,
            self.prefilter,
            self.gain,
            self.est,
            self.frame_ms,
            self.window,
        )


@dataclass(frozen=True)
class SystemOutput:
    enhanced: FloatArray
    speech_component: FloatArray | None


def system_to_dict(system: System) -> dict[str, Any]:
    return asdict(system)


def system_from_dict(data: dict[str, Any]) -> System:
    known = {f.name for f in fields(System)}
    fixed = {k: v for k, v in data.items() if k in known}
    for key in ("gain", "est"):
        fixed[key] = tuple((str(k), float(v)) for k, v in fixed.get(key, ()))
    return System(**fixed)


def config_for_frame_ms(frame_ms: int, base: Config = DEFAULT) -> Config:
    """Configuration for another frame length with all durations in seconds kept the same."""
    frame_len = frame_ms * base.fs // 1000
    if frame_len * 1000 != frame_ms * base.fs:
        raise ValueError("frame_ms must give an integer number of samples")
    hop = frame_len // 2
    exponent = hop / base.hop
    sub_frames = max(1, round(base.ms_sub_frames * base.hop / hop))
    return replace(
        base,
        frame_len=frame_len,
        hop=hop,
        ms_sub_frames=sub_frames,
        ms_alpha_s=base.ms_alpha_s**exponent,
        mcra_alpha_s=base.mcra_alpha_s**exponent,
        mcra_alpha_p=base.mcra_alpha_p**exponent,
        mcra_alpha_d=base.mcra_alpha_d**exponent,
        dd_alpha=base.dd_alpha**exponent,
        gain_mu=base.gain_mu**exponent,
    )


def _scaled(pairs: Pairs, keys: tuple[str, ...], exponent: float) -> dict[str, float]:
    return {k: (v**exponent if k in keys else v) for k, v in pairs}


def resolve(system: System) -> tuple[Config, dict[str, float], dict[str, float]]:
    """The configuration and the gain/estimator overrides a gain system actually runs with."""
    cfg = config_for_frame_ms(system.frame_ms)
    exponent = cfg.hop / REFERENCE_HOP
    gain = _scaled(system.gain, _RATE_KEYS_GAIN, exponent)
    est = _scaled(system.est, _RATE_KEYS_EST, exponent)
    if "sub_frames" in est:
        est["sub_frames"] = float(max(1, round(est["sub_frames"] / exponent)))
    if system.prefilter == "spectral":
        gain["spectral_hp"] = 1.0
    return cfg, gain, est


def run_system(
    system: System,
    clean: ArrayLike,
    noise: ArrayLike,
    noisy: ArrayLike,
    need_components: bool = True,
) -> SystemOutput:
    """Run one system on a mixture. ``noisy`` must equal ``clean + noise``.

    ``need_components=False`` skips the speech component (one STFT pass instead of three), for
    the tuning loops that only look at SNR.
    """
    y = np.asarray(noisy, dtype=np.float64)
    if system.kind == "unprocessed":
        return SystemOutput(y.copy(), None)
    if system.kind == "scipy_wiener":
        return SystemOutput(b2_scipy_wiener(y, system.window), None)
    if system.kind == "noisereduce":
        return SystemOutput(b3_noisereduce(y), None)
    cfg, gain, est = resolve(system)
    if not need_components:
        enhanced = enhance(
            y,
            cfg,
            rule=system.rule,
            estimator=system.estimator,
            prefilter=system.prefilter == "biquad",
            gain_overrides=gain,
            estimator_overrides=est,
        )
        return SystemOutput(enhanced, None)
    parts = gain_components(
        clean,
        noise,
        cfg,
        rule=system.rule,
        estimator=system.estimator,
        prefilter=system.prefilter == "biquad",
        gain_overrides=gain,
        estimator_overrides=est,
    )
    return SystemOutput(parts.output, parts.speech)


# --------------------------------------------------------------------------- registries
def b5_from(base: System) -> System:
    return replace(base, name="B5", label="HushWave (full)", external=True)


def make_baselines(b5: System, b1_gain: Pairs, b4_gain: Pairs, b2_window: int) -> list[System]:
    """B0 to B5. B1, B4 and B5 share the STFT, tracker, pre-filter and rise limit of B5, so they
    differ only in the gain rule (and the rule's own tuned parameters)."""
    return [
        System("B0", "unprocessed", "unprocessed", external=True),
        replace(b5, name="B1", label="spectral subtraction (G1)", rule="subtraction", gain=b1_gain),
        System("B2", "scipy.signal.wiener", "scipy_wiener", window=b2_window, external=True),
        System("B3", "noisereduce (default)", "noisereduce", external=True),
        replace(b5, name="B4", label="Wiener, plain SNR", rule="ml_wiener", gain=b4_gain),
        b5,
    ]


def make_ablations(b5: System) -> list[System]:
    """B5 with one component changed at a time, de-duplicated against B5 and each other."""
    gain = dict(b5.gain)
    candidates: list[System] = []

    def add(tag: str, label: str, system: System) -> None:
        candidates.append(replace(system, name=f"A:{tag}", label=label, external=False))

    for rule in ("wiener", "subtraction"):
        add(f"rule-{rule}", f"gain rule {rule}", replace(b5, rule=rule))
    other = "minstats" if b5.estimator == "mcra" else "mcra"
    add(f"est-{other}", f"noise tracker {other}", replace(b5, estimator=other))
    rising = dict(b5.est).get("rise_db_per_s", 0.0) > 0.0
    add(
        "rise-off" if rising else "rise-3",
        "no rise limit" if rising else "rise limit 3 dB/s",
        b5.with_est(rise_db_per_s=0.0 if rising else 3.0),
    )
    for mode in PREFILTERS:
        add(f"prefilter-{mode}", f"pre-filter {mode}", replace(b5, prefilter=mode))
    add("gain-smoothing-off", "no gain smoothing", b5.with_gain(gain_mu=0.0))
    add("weak-dd", "weak decision-directed memory", b5.with_gain(dd_alpha=0.5))
    base_floor = gain.get("g_min", DEFAULT.g_min)
    add("gmin-low", "lower gain floor", b5.with_gain(g_min=base_floor / 2.0))
    add("gmin-high", "higher gain floor", b5.with_gain(g_min=min(1.0, base_floor * 3.0)))
    for ms in (16, 64):
        add(f"frame-{ms}ms", f"frame length {ms} ms", replace(b5, frame_ms=ms))
    seen = {b5.structure()}
    unique: list[System] = []
    for system in candidates:
        if system.structure() not in seen:
            seen.add(system.structure())
            unique.append(system)
    return unique
