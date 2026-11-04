from __future__ import annotations

import math

import numpy as np
import pytest

from hushwave.stats import bootstrap_mean_ci, cluster_means, holm_adjust, paired_wilcoxon


def test_cluster_means_average_within_clusters_and_skip_nan() -> None:
    names, means = cluster_means([1.0, 3.0, np.nan, 10.0, np.nan], ["b", "b", "a", "c", "a"])
    assert names == ["b", "c"]
    np.testing.assert_allclose(means, [2.0, 10.0])
    with pytest.raises(ValueError):
        cluster_means([1.0, 2.0], ["a"])


def test_bootstrap_interval_contains_the_mean_and_is_deterministic() -> None:
    data = np.arange(20, dtype=float)
    first = bootstrap_mean_ci(data, seed=3)
    assert first == bootstrap_mean_ci(data, seed=3)
    assert first.lo < first.mean < first.hi
    assert first.mean == pytest.approx(9.5)
    assert first.n == 20
    wide = bootstrap_mean_ci(data, level=0.99, seed=3)
    assert wide.hi - wide.lo > first.hi - first.lo


def test_bootstrap_interval_covers_the_true_mean_about_95_percent() -> None:
    rng = np.random.default_rng(5)
    hits = 0
    for trial in range(200):
        sample = rng.normal(2.0, 1.0, 30)
        ci = bootstrap_mean_ci(sample, n_boot=400, seed=trial)
        hits += ci.lo <= 2.0 <= ci.hi
    assert 0.88 <= hits / 200 <= 0.99


def test_bootstrap_edge_cases() -> None:
    assert bootstrap_mean_ci([4.0]) == bootstrap_mean_ci([4.0, np.nan])
    assert math.isnan(bootstrap_mean_ci([np.nan]).mean)
    with pytest.raises(ValueError):
        bootstrap_mean_ci([1.0, 2.0], level=1.0)
    with pytest.raises(ValueError):
        bootstrap_mean_ci([1.0, 2.0], n_boot=0)


def test_wilcoxon_detects_a_shift_and_ignores_nan() -> None:
    rng = np.random.default_rng(1)
    b = rng.normal(0.0, 1.0, 40)
    a = b + 1.0 + 0.1 * rng.standard_normal(40)
    a[0] = np.nan
    result = paired_wilcoxon(a, b)
    assert result.n == 39
    assert result.p_value < 1e-6
    assert result.median_diff == pytest.approx(1.0, abs=0.2)
    same = paired_wilcoxon(b, b)
    assert (same.p_value, same.median_diff) == (1.0, 0.0)
    assert math.isnan(paired_wilcoxon([np.nan], [1.0]).p_value)
    with pytest.raises(ValueError):
        paired_wilcoxon([1.0, 2.0], [1.0])


def test_wilcoxon_p_value_is_roughly_uniform_under_the_null() -> None:
    rng = np.random.default_rng(2)
    p = [paired_wilcoxon(rng.normal(size=25), rng.normal(size=25)).p_value for _ in range(300)]
    assert 0.02 < np.mean(np.array(p) < 0.05) < 0.10


def test_holm_matches_a_hand_computation() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03, math.nan])
    assert adjusted[0] == pytest.approx(0.03)
    assert adjusted[2] == pytest.approx(0.06)
    assert adjusted[1] == pytest.approx(0.06)
    assert math.isnan(adjusted[3])
    assert holm_adjust([0.9, 0.8])[0] == 1.0
    assert holm_adjust([]) == []
