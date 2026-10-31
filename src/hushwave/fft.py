"""Radix-2 decimation-in-time FFT, written from first principles.

For N = 2^m the DFT X[k] = sum_n x[n] exp(-2 pi i n k / N) splits into the DFTs E and O of the
even- and odd-indexed samples:

    X[k]         = E[k] + w^k O[k]
    X[k + N / 2] = E[k] - w^k O[k],      w = exp(-2 pi i / N),  k = 0 .. N/2 - 1.

Applying this recursively, after a bit-reversal permutation of the input, gives log2(N) stages of
N / 2 butterflies each. All transforms act on the last axis and accept any leading batch shape,
so a whole spectrogram worth of frames is transformed in one call.
"""

from __future__ import annotations

from functools import cache

import numpy as np
from numpy.typing import ArrayLike, NDArray

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


def is_power_of_two(n: int) -> bool:
    """True when n is a positive power of two."""
    return n > 0 and n & (n - 1) == 0


def _check_length(n: int) -> None:
    if not is_power_of_two(n):
        raise ValueError(f"FFT length must be a power of two, got {n}")


@cache
def _plan(n: int) -> tuple[NDArray[np.intp], tuple[ComplexArray, ...]]:
    """Bit-reversal permutation and per-stage twiddle factors for length n."""
    bits = n.bit_length() - 1
    index = np.arange(n, dtype=np.intp)
    reverse = np.zeros(n, dtype=np.intp)
    for bit in range(bits):
        reverse |= ((index >> bit) & 1) << (bits - 1 - bit)
    twiddles = []
    size = 2
    while size <= n:
        k = np.arange(size // 2, dtype=np.float64)
        twiddle = np.exp(-2j * np.pi * k / size).astype(np.complex128)
        twiddle.setflags(write=False)
        twiddles.append(twiddle)
        size *= 2
    reverse.setflags(write=False)
    return reverse, tuple(twiddles)


def fft(x: ArrayLike) -> ComplexArray:
    """Discrete Fourier transform along the last axis (length must be a power of two)."""
    data = np.asarray(x, dtype=np.complex128)
    if data.ndim == 0:
        raise ValueError("fft needs at least a 1-D input")
    n = data.shape[-1]
    _check_length(n)
    if n == 1:
        return data.copy()
    reverse, twiddles = _plan(n)
    batch = data.shape[:-1]
    work = data[..., reverse]
    for twiddle in twiddles:
        half = twiddle.shape[0]
        blocks = work.reshape(*batch, n // (2 * half), 2, half)
        even = blocks[..., 0, :]
        odd = blocks[..., 1, :] * twiddle
        combined = np.empty_like(blocks)
        combined[..., 0, :] = even + odd
        combined[..., 1, :] = even - odd
        work = combined.reshape(*batch, n)
    return work


def ifft(spectrum: ArrayLike) -> ComplexArray:
    """Inverse DFT along the last axis, via conj(fft(conj(X))) / N."""
    data = np.asarray(spectrum, dtype=np.complex128)
    if data.ndim == 0:
        raise ValueError("ifft needs at least a 1-D input")
    n = data.shape[-1]
    _check_length(n)
    return np.asarray(np.conj(fft(np.conj(data))) / n, dtype=np.complex128)


def rfft(x: ArrayLike) -> ComplexArray:
    """One-sided DFT of a real signal: bins 0 .. N/2 (N/2 + 1 values)."""
    data = np.asarray(x, dtype=np.float64)
    if data.ndim == 0:
        raise ValueError("rfft needs at least a 1-D input")
    n = data.shape[-1]
    return fft(data)[..., : n // 2 + 1]


def irfft(spectrum: ArrayLike, n: int | None = None) -> FloatArray:
    """Inverse of rfft. The imaginary parts of the DC and Nyquist bins are ignored."""
    data = np.asarray(spectrum, dtype=np.complex128)
    if data.ndim == 0:
        raise ValueError("irfft needs at least a 1-D input")
    bins = data.shape[-1]
    length = 2 * (bins - 1) if n is None else n
    _check_length(length)
    if bins != length // 2 + 1:
        raise ValueError(f"a length-{length} signal has {length // 2 + 1} bins, got {bins}")
    mirrored = np.conj(data[..., 1 : length // 2][..., ::-1])
    full = np.concatenate([data, mirrored], axis=-1)
    return np.asarray(np.real(ifft(full)), dtype=np.float64)
