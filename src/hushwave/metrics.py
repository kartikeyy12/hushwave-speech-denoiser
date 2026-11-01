"""Evaluation metrics (docs/THEORY.md section 8.2).

Every metric except PESQ-WB and STOI is implemented here from its definition. PESQ-WB and STOI
come from the external packages ``pesq`` and ``pystoi`` and are labelled as such everywhere.

Segments. The clean reference is cut into non-overlapping blocks of ``cfg.hop`` samples (16 ms).
A block is speech-active when its mean square is within ``cfg.activity_db`` dB of the loudest
block; all other blocks are noise-only. Segmental SNR, speech distortion and the log-spectral
distance look at speech-active material; noise reduction and the musical-noise indicator look at
noise-only material. Read them together: a system that mutes everything scores well on noise
reduction and badly on the others.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import math
import warnings
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hushwave.config import DEFAULT, Config
from hushwave.stft import stft

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

SEG_SNR_MIN_DB = -10.0
SEG_SNR_MAX_DB = 35.0
LSD_FLOOR_DB = -100.0
MIN_NOISE_FRAMES = 20
PESQ_MIN_SECONDS = 0.25
EXTERNAL_METRICS = ("pesq_wb", "stoi")
_TINY = float(np.finfo(np.float64).tiny)


def _signal(x: ArrayLike, name: str) -> FloatArray:
    data = np.asarray(x, dtype=np.float64)
    if data.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {data.shape}")
    if data.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{name} contains NaN or Inf")
    return data


def _same_length(*pairs: tuple[str, ArrayLike]) -> list[FloatArray]:
    arrays = [_signal(value, name) for name, value in pairs]
    if any(a.shape != arrays[0].shape for a in arrays):
        names = ", ".join(name for name, _ in pairs)
        raise ValueError(f"{names} must have the same length")
    return arrays


# --------------------------------------------------------------------------- segments
def speech_activity(clean: ArrayLike, cfg: Config = DEFAULT) -> BoolArray:
    """Per-block speech-active flags from the clean reference (see the module docstring)."""
    x = _signal(clean, "clean")
    block = cfg.hop
    count = -(-x.size // block)
    padded = np.zeros(count * block)
    padded[: x.size] = x**2
    sums = padded.reshape(count, block).sum(axis=1)
    lengths = np.full(count, float(block))
    lengths[-1] = x.size - (count - 1) * block
    mean_square = sums / lengths
    peak = float(mean_square.max())
    if peak == 0.0:
        raise ValueError("clean reference is all zeros; speech activity is undefined")
    return np.asarray(mean_square >= peak * 10.0 ** (-cfg.activity_db / 10.0), dtype=np.bool_)


def _sample_mask(active: BoolArray, n_samples: int, block: int) -> BoolArray:
    return np.asarray(np.repeat(active, block)[:n_samples], dtype=np.bool_)


def active_fraction(active: BoolArray) -> float:
    """Fraction of blocks that are speech-active."""
    return float(np.mean(active))


# --------------------------------------------------------------------------- waveform SNR
def snr_db(clean: ArrayLike, estimate: ArrayLike) -> float:
    """10 log10( sum clean^2 / sum (estimate - clean)^2 ); +inf when the estimate is exact."""
    x, e = _same_length(("clean", clean), ("estimate", estimate))
    signal = float(np.sum(x**2))
    if signal == 0.0:
        raise ValueError("clean reference is all zeros; SNR is undefined")
    error = float(np.sum((e - x) ** 2))
    if error == 0.0:
        return math.inf
    return 10.0 * math.log10(signal / error)


def delta_snr_db(clean: ArrayLike, noisy: ArrayLike, enhanced: ArrayLike) -> float:
    """SNR of the enhanced signal minus SNR of the noisy signal, both against the clean one."""
    return snr_db(clean, enhanced) - snr_db(clean, noisy)


def segmental_snr_db(
    clean: ArrayLike,
    estimate: ArrayLike,
    cfg: Config = DEFAULT,
    lo_db: float = SEG_SNR_MIN_DB,
    hi_db: float = SEG_SNR_MAX_DB,
) -> float:
    """Mean over speech-active blocks of the block SNR, each clamped to [lo_db, hi_db]."""
    x, e = _same_length(("clean", clean), ("estimate", estimate))
    active = speech_activity(x, cfg)
    block = cfg.hop
    values = []
    for index in np.flatnonzero(active):
        s = slice(index * block, (index + 1) * block)
        signal = float(np.sum(x[s] ** 2))
        error = max(float(np.sum((e[s] - x[s]) ** 2)), _TINY)
        values.append(min(max(10.0 * math.log10(signal / error), lo_db), hi_db))
    return float(np.mean(values))


# --------------------------------------------------------------------------- noise and speech
def noise_reduction_db(
    clean: ArrayLike, noisy: ArrayLike, enhanced: ArrayLike, cfg: Config = DEFAULT
) -> float:
    """Energy of the noisy signal over energy of the enhanced signal on noise-only blocks, dB.

    NaN when the reference has no noise-only block.
    """
    x, y, z = _same_length(("clean", clean), ("noisy", noisy), ("enhanced", enhanced))
    quiet = ~_sample_mask(speech_activity(x, cfg), x.size, cfg.hop)
    if not np.any(quiet):
        return math.nan
    before = float(np.sum(y[quiet] ** 2))
    after = max(float(np.sum(z[quiet] ** 2)), _TINY)
    if before == 0.0:
        return math.nan
    return 10.0 * math.log10(before / after)


def speech_distortion_db(
    clean: ArrayLike, speech_component: ArrayLike, cfg: Config = DEFAULT
) -> float:
    """10 log10( sum (s' - s)^2 / sum s^2 ) on speech-active blocks; lower is better.

    s is the clean speech and s' is the speech component of the system output (the clean speech
    passed through the same time-varying gains), see ``enhancer.gain_components``.
    """
    x, s = _same_length(("clean", clean), ("speech_component", speech_component))
    loud = _sample_mask(speech_activity(x, cfg), x.size, cfg.hop)
    reference = float(np.sum(x[loud] ** 2))
    error = max(float(np.sum((s[loud] - x[loud]) ** 2)), _TINY)
    return 10.0 * math.log10(error / reference)


def _frame_blocks(n_frames: int, n_samples: int, cfg: Config) -> list[range]:
    """Blocks of cfg.hop samples overlapped by each STFT frame (frame f starts at f H - (N - H))."""
    lead = cfg.frame_len - cfg.hop
    last_block = (n_samples - 1) // cfg.hop
    out = []
    for f in range(n_frames):
        start = f * cfg.hop - lead
        first = max(start, 0) // cfg.hop
        last = min(start + cfg.frame_len - 1, n_samples - 1) // cfg.hop
        out.append(range(min(first, last_block), min(last, last_block) + 1))
    return out


def lsd_db(
    clean: ArrayLike, estimate: ArrayLike, cfg: Config = DEFAULT, floor_db: float = LSD_FLOOR_DB
) -> float:
    """Log-spectral distance between clean and estimate over frames that touch speech, in dB.

    Per frame: sqrt( mean over bins of (10 log10 P_clean - 10 log10 P_estimate)^2 ), with both
    powers floored at floor_db below the loudest clean bin.
    """
    x, e = _same_length(("clean", clean), ("estimate", estimate))
    active = speech_activity(x, cfg)
    power_x = np.abs(stft(x, cfg.frame_len, cfg.hop)) ** 2
    power_e = np.abs(stft(e, cfg.frame_len, cfg.hop)) ** 2
    floor = float(power_x.max()) * 10.0 ** (floor_db / 10.0)
    blocks = _frame_blocks(power_x.shape[0], x.size, cfg)
    keep = np.array([bool(np.any(active[list(r)])) for r in blocks])
    diff = 10.0 * np.log10(power_x[keep] + floor) - 10.0 * np.log10(power_e[keep] + floor)
    return float(np.mean(np.sqrt(np.mean(diff**2, axis=1))))


def musical_noise_ratio(
    clean: ArrayLike, noisy: ArrayLike, enhanced: ArrayLike, cfg: Config = DEFAULT
) -> float:
    """Kurtosis ratio of the noise-only power spectra, output over input (Uemura et al. style).

    For every frequency bin the kurtosis of the power over time is mean(P^2) / mean(P)^2, taken
    over STFT frames that lie entirely in noise-only blocks. The indicator is the mean kurtosis of
    the enhanced signal divided by that of the noisy signal. Gaussian noise has kurtosis 2, so
    1 means "no more peaky than the input"; clearly above 1 means isolated surviving spectral
    peaks, the cause of musical noise. NaN when fewer than MIN_NOISE_FRAMES frames qualify.
    """
    x, y, z = _same_length(("clean", clean), ("noisy", noisy), ("enhanced", enhanced))
    active = speech_activity(x, cfg)
    power_y = np.abs(stft(y, cfg.frame_len, cfg.hop)) ** 2
    power_z = np.abs(stft(z, cfg.frame_len, cfg.hop)) ** 2
    blocks = _frame_blocks(power_y.shape[0], x.size, cfg)
    keep = np.array([not bool(np.any(active[list(r)])) for r in blocks])
    if int(keep.sum()) < MIN_NOISE_FRAMES:
        return math.nan

    def kurtosis(power: FloatArray) -> float:
        mean = power.mean(axis=0)
        second = (power**2).mean(axis=0)
        valid = mean > 0.0
        if not np.any(valid):
            return math.nan
        return float(np.mean(second[valid] / mean[valid] ** 2))

    return kurtosis(power_z[keep]) / kurtosis(power_y[keep])


# --------------------------------------------------------------------------- external metrics
def _external(name: str) -> object:
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise ImportError(
            f"{name} is an optional external package: pip install -e '.[metrics]'"
        ) from error


def external_versions() -> dict[str, str]:
    """Versions of the external metric packages, for the report (empty string if missing)."""
    versions = {}
    for package in ("pesq", "pystoi"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = ""
    return versions


def pesq_wb(reference: ArrayLike, degraded: ArrayLike, fs: int = DEFAULT.fs) -> float:
    """PESQ wide-band (ITU-T P.862.2) from the external `pesq` package; needs fs = 16000.

    NaN if either signal is silent or the package finds no utterance. Raises ValueError for
    signals shorter than a quarter of a second or an unsupported sample rate.
    """
    ref, deg = _same_length(("reference", reference), ("degraded", degraded))
    if fs != 16_000:
        raise ValueError("PESQ wide-band needs fs = 16000 Hz")
    if ref.size < PESQ_MIN_SECONDS * fs:
        raise ValueError(f"PESQ needs at least {PESQ_MIN_SECONDS} s of audio")
    if not (np.any(ref) and np.any(deg)):
        return math.nan
    module = _external("pesq")
    try:
        return float(module.pesq(fs, ref, deg, "wb"))  # type: ignore[attr-defined]
    except Exception as error:
        if type(error).__name__ == "NoUtterancesError":
            return math.nan
        raise


def stoi_score(
    reference: ArrayLike, degraded: ArrayLike, fs: int = DEFAULT.fs, extended: bool = False
) -> float:
    """STOI (or ESTOI) from the external `pystoi` package, which works at 10 kHz internally.

    NaN when pystoi reports too few speech frames (it would return 1e-5 with a warning).
    """
    ref, deg = _same_length(("reference", reference), ("degraded", degraded))
    module = _external("pystoi")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = float(module.stoi(ref, deg, fs, extended=extended))  # type: ignore[attr-defined]
    if any("Not enough STFT frames" in str(w.message) for w in caught):
        return math.nan
    return value


# --------------------------------------------------------------------------- everything at once
@dataclass(frozen=True)
class Metrics:
    """All metrics for one enhanced signal. NaN means "not defined for this input"."""

    snr_in_db: float
    snr_out_db: float
    delta_snr_db: float
    segsnr_in_db: float
    segsnr_out_db: float
    delta_segsnr_db: float
    lsd_in_db: float
    lsd_out_db: float
    noise_reduction_db: float
    speech_distortion_db: float
    musical_noise_ratio: float
    pesq_in: float
    pesq_out: float
    stoi_in: float
    stoi_out: float
    active_fraction: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def evaluate(
    clean: ArrayLike,
    noisy: ArrayLike,
    enhanced: ArrayLike,
    speech_component: ArrayLike | None = None,
    cfg: Config = DEFAULT,
    external: bool = True,
) -> Metrics:
    """Every metric for one (clean, noisy, enhanced) triple.

    speech_component is the clean speech after the system's own time-varying gains; without it
    (systems that are not gain-based) the speech distortion is NaN. external=False skips PESQ and
    STOI, which are the slow ones and need the optional packages.
    """
    x, y, z = _same_length(("clean", clean), ("noisy", noisy), ("enhanced", enhanced))
    seg_in = segmental_snr_db(x, y, cfg)
    seg_out = segmental_snr_db(x, z, cfg)
    distortion = (
        math.nan if speech_component is None else speech_distortion_db(x, speech_component, cfg)
    )
    pesq_in = pesq_out = stoi_in = stoi_out = math.nan
    if external:
        pesq_in, pesq_out = pesq_wb(x, y, cfg.fs), pesq_wb(x, z, cfg.fs)
        stoi_in, stoi_out = stoi_score(x, y, cfg.fs), stoi_score(x, z, cfg.fs)
    return Metrics(
        snr_in_db=snr_db(x, y),
        snr_out_db=snr_db(x, z),
        delta_snr_db=delta_snr_db(x, y, z),
        segsnr_in_db=seg_in,
        segsnr_out_db=seg_out,
        delta_segsnr_db=seg_out - seg_in,
        lsd_in_db=lsd_db(x, y, cfg),
        lsd_out_db=lsd_db(x, z, cfg),
        noise_reduction_db=noise_reduction_db(x, y, z, cfg),
        speech_distortion_db=distortion,
        musical_noise_ratio=musical_noise_ratio(x, y, z, cfg),
        pesq_in=pesq_in,
        pesq_out=pesq_out,
        stoi_in=stoi_in,
        stoi_out=stoi_out,
        active_fraction=active_fraction(speech_activity(x, cfg)),
    )
