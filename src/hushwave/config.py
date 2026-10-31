"""Project-wide parameters.

Every value here is justified numerically in docs/THEORY.md, and the numbers quoted in that
document are computed from this module by scripts/fill_docs.py (never typed by hand).
Later milestones add the parameters they introduce, each with its own justification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Signal, framing, pre-filter and estimator-memory parameters."""

    fs: int = 16_000
    pcm_bits: int = 16
    frame_len: int = 512
    hop: int = 256
    hp_cutoff_hz: float = 80.0
    mains_hz: float = 50.0
    ms_n_sub: int = 6
    ms_sub_frames: int = 16
    ms_alpha_s: float = 0.9
    mcra_alpha_s: float = 0.8
    mcra_alpha_p: float = 0.2
    mcra_alpha_d: float = 0.95
    mcra_delta: float = 5.0
    noise_floor: float = 1e-12
    dd_alpha: float = 0.98

    def __post_init__(self) -> None:
        if self.fs <= 0:
            raise ValueError(f"fs must be positive, got {self.fs}")
        if not 8 <= self.pcm_bits <= 32:
            raise ValueError(f"pcm_bits must be in [8, 32], got {self.pcm_bits}")
        if self.frame_len <= 0 or self.frame_len & (self.frame_len - 1) != 0:
            raise ValueError(f"frame_len must be a power of two, got {self.frame_len}")
        if 2 * self.hop != self.frame_len:
            raise ValueError("hop must be frame_len / 2 (50 % overlap, sqrt-Hann WOLA)")
        if not 0.0 < self.hp_cutoff_hz < self.fs / 2:
            raise ValueError("hp_cutoff_hz must lie strictly between 0 and Nyquist")
        if self.mains_hz <= 0.0:
            raise ValueError("mains_hz must be positive")
        if self.ms_n_sub < 1 or self.ms_sub_frames < 1:
            raise ValueError("minimum-statistics window dimensions must be at least 1")
        for name in ("dd_alpha", "ms_alpha_s", "mcra_alpha_s", "mcra_alpha_p", "mcra_alpha_d"):
            if not 0.0 < getattr(self, name) < 1.0:
                raise ValueError(f"{name} must lie strictly between 0 and 1")
        if self.mcra_delta <= 1.0:
            raise ValueError("mcra_delta must exceed 1")
        if self.noise_floor <= 0.0:
            raise ValueError("noise_floor must be positive")

    @property
    def nyquist_hz(self) -> float:
        return self.fs / 2

    @property
    def frame_ms(self) -> float:
        return 1000.0 * self.frame_len / self.fs

    @property
    def hop_ms(self) -> float:
        return 1000.0 * self.hop / self.fs

    @property
    def bin_hz(self) -> float:
        return self.fs / self.frame_len

    @property
    def n_bins(self) -> int:
        return self.frame_len // 2 + 1

    @property
    def ms_window_frames(self) -> int:
        return self.ms_n_sub * self.ms_sub_frames

    @property
    def ms_window_s(self) -> float:
        return self.ms_window_frames * self.hop / self.fs

    @property
    def latency_min_samples(self) -> int:
        """Smallest per-sample algorithmic delay: frame_len - hop (derived in THEORY.md)."""
        return self.frame_len - self.hop

    @property
    def latency_max_samples(self) -> int:
        """Largest per-sample algorithmic delay: frame_len - 1 (derived in THEORY.md)."""
        return self.frame_len - 1

    @property
    def dd_time_constant_ms(self) -> float:
        """Time constant of the decision-directed recursion, hop / -ln(alpha)."""
        return -self.hop_ms / math.log(self.dd_alpha)


DEFAULT = Config()
