"""Evaluation of real recordings that have no clean reference (THEORY 8.3, kept separate).

For "speech in fan" and "speech in corridor" the clean speech is unknown, so no SNR, PESQ or STOI
can be computed. What can be measured is what the system does to the pauses and to the speech:

* A block of ``hop`` samples is a *pause* when its energy does not exceed the 95th percentile of
  the block energies of the matching noise-only recording, plus a margin; it is *speech* when its
  energy exceeds that floor by ``speech_margin_db``. Blocks in between are left out of both.
* noise reduction on pauses, residual level in pauses, musical-noise ratio on pauses,
* change of the speech-block energy (near 0 dB means the speech level was kept; the energy
  includes the noise that sits under the speech, so a small negative value is expected),
* the Welch PSD (our own implementation) of the pauses before and after, next to the PSD of the
  noise-only recording, which shows what noise the system removed and whether it kept its colour.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hushwave.audio_io import write_wav
from hushwave.config import DEFAULT, Config
from hushwave.metrics import musical_noise_masked, noise_reduction_masked
from hushwave.spectral import welch
from hushwave.systems import System, run_system

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
PSD_NPERSEG = 512
MIN_RUN_BLOCKS = 2 * PSD_NPERSEG // DEFAULT.hop
FLOOR_PERCENTILE = 95.0
PAUSE_MARGIN_DB = 1.0
SPEECH_MARGIN_DB = 10.0
TINY = 1e-20


@dataclass(frozen=True)
class Segmentation:
    pause: BoolArray
    speech: BoolArray
    floor_db: float


def block_energy_db(x: ArrayLike, block: int) -> FloatArray:
    """Mean-square energy of each complete block, in dB (full-scale sine of amplitude 1: -3 dB)."""
    data = np.asarray(x, dtype=np.float64)
    count = data.size // block
    if count < 1:
        raise ValueError("signal shorter than one block")
    energy = np.mean(data[: count * block].reshape(count, block) ** 2, axis=1)
    return np.asarray(10.0 * np.log10(energy + TINY), dtype=np.float64)


def segment_by_noise_floor(
    noisy: ArrayLike,
    noise_reference: ArrayLike | None,
    cfg: Config = DEFAULT,
    pause_margin_db: float = PAUSE_MARGIN_DB,
    speech_margin_db: float = SPEECH_MARGIN_DB,
) -> Segmentation:
    """Pause and speech blocks of ``noisy``, using the noise-only recording as the yardstick.

    Without a noise reference the floor is the 10th percentile of the noisy block energies plus
    6 dB, a rougher rule that is reported as such by the caller.
    """
    energy = block_energy_db(noisy, cfg.hop)
    if noise_reference is not None:
        floor = float(np.percentile(block_energy_db(noise_reference, cfg.hop), FLOOR_PERCENTILE))
    else:
        floor = float(np.percentile(energy, 10.0)) + 6.0
    # Same convention as metrics.speech_activity: one flag per block, the partial last block
    # included. It is never counted as pause or speech (too short to judge).
    tail = np.zeros(-(-np.asarray(noisy).size // cfg.hop) - energy.size, dtype=np.bool_)
    return Segmentation(
        pause=np.concatenate([energy <= floor + pause_margin_db, tail]),
        speech=np.concatenate([energy > floor + speech_margin_db, tail]),
        floor_db=floor,
    )


def _runs(flags: BoolArray, minimum: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(list(flags) + [False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if index - start >= minimum:
                runs.append((start, index))
            start = None
    return runs


def pause_psd(
    x: ArrayLike, pause: BoolArray, cfg: Config = DEFAULT
) -> tuple[FloatArray, FloatArray] | None:
    """Welch PSD averaged over all pause runs of at least 2 * nperseg samples; None if none."""
    data = np.asarray(x, dtype=np.float64)
    total: FloatArray | None = None
    weight = 0.0
    freqs = np.zeros(0)
    for first, last in _runs(pause, MIN_RUN_BLOCKS):
        piece = data[first * cfg.hop : last * cfg.hop]
        freqs, psd = welch(piece, cfg.fs, PSD_NPERSEG)
        w = float(piece.size)
        total = psd * w if total is None else total + psd * w
        weight += w
    if total is None:
        return None
    return freqs, np.asarray(total / weight, dtype=np.float64)


def _db(value: float) -> float:
    return 10.0 * math.log10(max(value, TINY))


def nonintrusive_metrics(
    noisy: ArrayLike, enhanced: ArrayLike, seg: Segmentation, cfg: Config = DEFAULT
) -> dict[str, float]:
    """Metrics of one enhanced recording; NaN where the segmentation leaves nothing to measure."""
    y = np.asarray(noisy, dtype=np.float64)
    z = np.asarray(enhanced, dtype=np.float64)
    mask_pause = np.repeat(seg.pause, cfg.hop)[: y.size]
    mask_speech = np.repeat(seg.speech, cfg.hop)[: y.size]
    pad = y.size - mask_pause.size
    if pad > 0:
        mask_pause = np.concatenate([mask_pause, np.zeros(pad, dtype=bool)])
        mask_speech = np.concatenate([mask_speech, np.zeros(pad, dtype=bool)])
    speech_change = math.nan
    if np.any(mask_speech) and np.sum(y[mask_speech] ** 2) > 0.0:
        speech_change = _db(float(np.mean(z[mask_speech] ** 2))) - _db(
            float(np.mean(y[mask_speech] ** 2))
        )
    resid = math.nan
    if np.any(mask_pause):
        resid = _db(float(np.mean(z[mask_pause] ** 2)))
    return {
        "noise_reduction_pause_db": noise_reduction_masked(y, z, seg.pause, cfg),
        "residual_pause_dbfs": resid,
        "speech_level_change_db": speech_change,
        "musical_noise_ratio": musical_noise_masked(y, z, seg.pause, cfg),
    }


def evaluate_recording(
    noisy: ArrayLike,
    noise_reference: ArrayLike | None,
    systems: Sequence[System],
    cfg: Config = DEFAULT,
    save_dir: Path | None = None,
    stem: str = "recording",
    psd_systems: Sequence[str] = ("B0", "B1", "B5"),
) -> dict[str, Any]:
    """Run every system on a real recording and measure pauses and speech without a reference."""
    y = np.asarray(noisy, dtype=np.float64)
    seg = segment_by_noise_floor(y, noise_reference, cfg)
    result: dict[str, Any] = {
        "duration_s": y.size / cfg.fs,
        "noise_floor_db": seg.floor_db,
        "floor_from_noise_recording": noise_reference is not None,
        "pause_blocks": int(seg.pause.sum()),
        "speech_blocks": int(seg.speech.sum()),
        "total_blocks": int(seg.pause.size),
        "systems": {},
    }
    psd: dict[str, Any] = {}
    outputs: dict[str, FloatArray] = {}
    for system in systems:
        z = run_system(system, y, np.zeros_like(y), y, need_components=False).enhanced
        outputs[system.name] = z
        result["systems"][system.name] = nonintrusive_metrics(y, z, seg, cfg)
        if save_dir is not None and system.name != "B0":
            save_dir.mkdir(parents=True, exist_ok=True)
            write_wav(save_dir / f"{stem}_{system.name}.wav", z, cfg.fs)
    for name in psd_systems:
        if name in outputs:
            pair = pause_psd(outputs[name], seg.pause, cfg)
            if pair is not None:
                psd["freq_hz"] = pair[0]
                psd[f"{name}_db"] = 10.0 * np.log10(pair[1] + TINY)
    if noise_reference is not None:
        ref = np.asarray(noise_reference, dtype=np.float64)
        if ref.size >= PSD_NPERSEG:
            freqs, ref_psd = welch(ref, cfg.fs, PSD_NPERSEG)
            psd["freq_hz"] = freqs
            psd["noise_reference_db"] = 10.0 * np.log10(ref_psd + TINY)
    result["psd"] = psd
    return result
