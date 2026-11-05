"""Live and replay support for the demo (M9).

The demo runs the frozen B5 settings from ``results/tuning.json`` on the streaming engine of
``enhancer.py``. Nothing is retuned here. ``LiveSession`` wraps one engine for a block-by-block
audio stream: it keeps the engine running while enhancement is switched off (so its noise
estimate stays current), cross-fades at the switch, and reports level and compute time per block.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hushwave.enhancer import StreamingEnhancer, enhance
from hushwave.systems import (
    Pairs,
    System,
    make_baselines,
    resolve,
    system_from_dict,
)

FloatArray = NDArray[np.float64]
ROOT = Path(__file__).resolve().parents[2]
TUNING_PATH = ROOT / "results" / "tuning.json"
LEVEL_FLOOR_DB = -90.0
MAX_BLOCK = 8192


def level_dbfs(x: ArrayLike) -> float:
    """RMS level of a block in dB relative to full scale, floored at LEVEL_FLOOR_DB."""
    data = np.asarray(x, dtype=np.float64)
    if data.size == 0:
        return LEVEL_FLOOR_DB
    rms = float(np.sqrt(np.mean(data**2)))
    return LEVEL_FLOOR_DB if rms <= 0.0 else max(LEVEL_FLOOR_DB, 20.0 * math.log10(rms))


def _pairs(values: object) -> Pairs:
    if not isinstance(values, list):
        raise ValueError("tuning file: expected a list of [name, value] pairs")
    return tuple((str(k), float(v)) for k, v in values)


def load_tuning(path: Path = TUNING_PATH) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run the M8 tuning stage first")
    data: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    return data


def load_b5(path: Path = TUNING_PATH) -> System:
    """The frozen B5 system exactly as the TEST evaluation used it."""
    final = load_tuning(path)["final"]
    if not isinstance(final, dict):
        raise ValueError("tuning file: 'final' must be an object")
    return system_from_dict(final["b5"])


def load_b1(path: Path = TUNING_PATH) -> System:
    """B1 (spectral subtraction) on the same pipeline, with its own tuned parameters."""
    final = load_tuning(path)["final"]
    if not isinstance(final, dict):
        raise ValueError("tuning file: 'final' must be an object")
    b5 = system_from_dict(final["b5"])
    baselines = make_baselines(b5, _pairs(final["b1_gain"]), _pairs(final["b4_gain"]), 15)
    return next(s for s in baselines if s.name == "B1")


def build_engine(system: System) -> StreamingEnhancer:
    """A streaming engine configured exactly like the batch run of ``system``."""
    if system.kind != "gain":
        raise ValueError("only gain-based systems run on the streaming engine")
    cfg, gain, est = resolve(system)
    return StreamingEnhancer(
        cfg,
        rule=system.rule,
        estimator=system.estimator,
        prefilter=system.prefilter == "biquad",
        gain_overrides=gain,
        estimator_overrides=est,
    )


def render(system: System, x: ArrayLike) -> FloatArray:
    """Process a whole signal in one pass (equal to the streamed result, section 4.5)."""
    cfg, gain, est = resolve(system)
    return enhance(
        np.asarray(x, dtype=np.float64),
        cfg,
        rule=system.rule,
        estimator=system.estimator,
        prefilter=system.prefilter == "biquad",
        gain_overrides=gain,
        estimator_overrides=est,
    )


@dataclass(frozen=True)
class BlockStats:
    level_in_dbfs: float
    level_out_dbfs: float
    compute_ms: float


class LiveSession:
    """One audio stream through the engine, block by block."""

    def __init__(self, system: System) -> None:
        self.system = system
        self.engine = build_engine(system)
        self.cfg = self.engine.cfg
        self.delay_samples = self.engine.delay_samples
        self._enhanced = True
        self.blocks = 0
        self.samples = 0

    @property
    def fs(self) -> int:
        return int(self.cfg.fs)

    @property
    def delay_ms(self) -> float:
        return 1000.0 * self.delay_samples / self.fs

    def process(self, block: ArrayLike, enhanced: bool = True) -> tuple[FloatArray, BlockStats]:
        """Process one block; the output has exactly as many samples as the input.

        The engine always runs, so its estimates stay current while enhancement is off. When
        ``enhanced`` changes, the output is cross-faded linearly over this block between the
        raw input (no delay) and the engine output (constant delay).
        """
        dry = np.asarray(block, dtype=np.float64)
        if dry.ndim != 1 or dry.size == 0 or dry.size > MAX_BLOCK:
            raise ValueError(f"block must be 1-D with 1 to {MAX_BLOCK} samples")
        if not np.all(np.isfinite(dry)):
            raise ValueError("block contains NaN or Inf")
        start = time.perf_counter()
        wet = self.engine.process(dry)
        compute_ms = 1000.0 * (time.perf_counter() - start)
        weight = np.full(dry.size, 1.0 if enhanced else 0.0)
        if enhanced != self._enhanced:
            ramp = (np.arange(dry.size) + 1.0) / dry.size
            weight = ramp if enhanced else 1.0 - ramp
        self._enhanced = enhanced
        out = np.clip(weight * wet + (1.0 - weight) * dry, -1.0, 1.0)
        self.blocks += 1
        self.samples += dry.size
        return out, BlockStats(level_dbfs(dry), level_dbfs(out), compute_ms)
