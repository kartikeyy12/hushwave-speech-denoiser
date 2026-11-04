"""Statistics for the experiments (docs/THEORY.md section 8.3).

Test mixtures are not independent: every utterance appears once per noise and per SNR. Treating
the cases as independent would make confidence intervals far too narrow, so the unit of analysis
is the utterance. Per-case values are first averaged within each utterance (``cluster_means``),
then the mean over utterances gets a percentile-bootstrap interval (resampling utterances), and
paired tests compare per-utterance means. ``scipy.stats.wilcoxon`` is the reference implementation
of the signed-rank test (external, labelled as such); the Holm step-down correction is ours.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import wilcoxon

FloatArray = NDArray[np.float64]
ALPHA = 0.05  # significance level used after Holm correction


@dataclass(frozen=True)
class MeanCI:
    """Mean of per-cluster means with a percentile-bootstrap confidence interval."""

    mean: float
    lo: float
    hi: float
    n: int


@dataclass(frozen=True)
class PairedTest:
    """Paired comparison a - b: sample size, median/mean difference, Wilcoxon p-value."""

    n: int
    median_diff: float
    mean_diff: float
    p_value: float


def cluster_means(values: ArrayLike, clusters: Sequence[str]) -> tuple[list[str], FloatArray]:
    """Mean of ``values`` within each cluster (NaN ignored); all-NaN clusters are dropped.

    Clusters are returned in sorted order, so two calls on the same cases line up.
    """
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or data.size != len(clusters):
        raise ValueError("values and clusters must be 1-D and of equal length")
    names: list[str] = []
    means: list[float] = []
    for name in sorted(set(clusters)):
        member = data[np.array([c == name for c in clusters])]
        member = member[np.isfinite(member)]
        if member.size:
            names.append(name)
            means.append(float(member.mean()))
    return names, np.asarray(means, dtype=np.float64)


def bootstrap_mean_ci(
    cluster_values: ArrayLike, n_boot: int = 2000, level: float = 0.95, seed: int = 0
) -> MeanCI:
    """Mean with a percentile bootstrap interval over the given cluster values."""
    data = np.asarray(cluster_values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return MeanCI(math.nan, math.nan, math.nan, 0)
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between 0 and 1")
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    mean = float(data.mean())
    if data.size == 1:
        return MeanCI(mean, mean, mean, 1)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, data.size, size=(n_boot, data.size))
    boot = data[picks].mean(axis=1)
    tail = 100.0 * (1.0 - level) / 2.0
    lo, hi = np.percentile(boot, [tail, 100.0 - tail])
    return MeanCI(mean, float(lo), float(hi), int(data.size))


def paired_wilcoxon(a: ArrayLike, b: ArrayLike) -> PairedTest:
    """Two-sided Wilcoxon signed-rank test on the pairs where both values are finite.

    All-equal pairs give p = 1; no usable pair gives NaN.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("a and b must be 1-D arrays of equal length")
    keep = np.isfinite(x) & np.isfinite(y)
    diff = x[keep] - y[keep]
    if diff.size == 0:
        return PairedTest(0, math.nan, math.nan, math.nan)
    if not np.any(diff != 0.0):
        return PairedTest(int(diff.size), 0.0, 0.0, 1.0)
    p_value = float(wilcoxon(diff).pvalue)
    return PairedTest(int(diff.size), float(np.median(diff)), float(diff.mean()), p_value)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1); NaN stays NaN."""
    indexed = [(p, i) for i, p in enumerate(p_values) if not math.isnan(p)]
    indexed.sort()
    adjusted = [math.nan] * len(p_values)
    running = 0.0
    count = len(indexed)
    for rank, (p, i) in enumerate(indexed):
        running = max(running, min(1.0, (count - rank) * p))
        adjusted[i] = running
    return adjusted
