"""Second-order IIR section (biquad) designed with the bilinear transform.

Analog prototype H(s) = s^2 / (s^2 + s / Q + 1). With K = tan(pi fc / fs) and
s = (1 - z^-1) / (K (1 + z^-1)) (the pre-warped bilinear transform), docs/THEORY.md section 5
derives

    b = [1, -2, 1] / D,   a = [1, 2 (K^2 - 1) / D, (1 - K / Q + K^2) / D],   D = 1 + K / Q + K^2.

The filter runs as the transposed direct form II recursion
    y[n] = b0 x[n] + z1,   z1 <- b1 x[n] - a1 y[n] + z2,   z2 <- b2 x[n] - a2 y[n].
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

BUTTERWORTH_Q = 1.0 / math.sqrt(2.0)


def highpass_biquad(
    cutoff_hz: float, fs: float, q: float = BUTTERWORTH_Q
) -> tuple[FloatArray, FloatArray]:
    """Coefficients (b, a), with a[0] = 1, of a 2nd-order high-pass at `cutoff_hz`."""
    if fs <= 0.0 or q <= 0.0:
        raise ValueError("fs and q must be positive")
    if not 0.0 < cutoff_hz < fs / 2:
        raise ValueError("cutoff_hz must lie strictly between 0 and Nyquist")
    k = math.tan(math.pi * cutoff_hz / fs)
    norm = 1.0 + k / q + k * k
    b = np.array([1.0, -2.0, 1.0]) / norm
    a = np.array([1.0, 2.0 * (k * k - 1.0) / norm, (1.0 - k / q + k * k) / norm])
    return b, a


class Biquad:
    """Stateful biquad: blocks of any length can be fed one after another."""

    def __init__(self, b: ArrayLike, a: ArrayLike) -> None:
        b_arr = np.asarray(b, dtype=np.float64)
        a_arr = np.asarray(a, dtype=np.float64)
        if b_arr.shape != (3,) or a_arr.shape != (3,):
            raise ValueError("b and a must each have 3 coefficients")
        if a_arr[0] == 0.0:
            raise ValueError("a[0] must be non-zero")
        scale = a_arr[0]
        self.b: FloatArray = b_arr / scale
        self.a: FloatArray = a_arr / scale
        self._z1 = 0.0
        self._z2 = 0.0

    def reset(self) -> None:
        """Clear the filter memory."""
        self._z1 = 0.0
        self._z2 = 0.0

    def process(self, x: ArrayLike) -> FloatArray:
        """Filter a block, carrying the state over to the next call."""
        data = np.asarray(x, dtype=np.float64)
        if data.ndim != 1:
            raise ValueError(f"expected a 1-D block, got shape {data.shape}")
        if not np.all(np.isfinite(data)):
            raise ValueError("block contains NaN or Inf")
        b0, b1, b2 = (float(v) for v in self.b)
        a1, a2 = float(self.a[1]), float(self.a[2])
        z1, z2 = self._z1, self._z2
        out: list[float] = []
        append = out.append
        for sample in data.tolist():
            y = b0 * sample + z1
            z1 = b1 * sample - a1 * y + z2
            z2 = b2 * sample - a2 * y
            append(y)
        self._z1, self._z2 = z1, z2
        return np.asarray(out, dtype=np.float64)


def frequency_response(b: ArrayLike, a: ArrayLike, freqs_hz: ArrayLike, fs: float) -> ComplexArray:
    """H(e^{j w}) = B(e^{j w}) / A(e^{j w}) at the given frequencies."""
    b_arr = np.asarray(b, dtype=np.float64)
    a_arr = np.asarray(a, dtype=np.float64)
    omega = 2.0 * np.pi * np.asarray(freqs_hz, dtype=np.float64) / fs
    taps = np.arange(b_arr.size)
    numerator = np.exp(-1j * np.outer(omega, taps)) @ b_arr
    denominator = np.exp(-1j * np.outer(omega, np.arange(a_arr.size))) @ a_arr
    return np.asarray(numerator / denominator, dtype=np.complex128)


def pole_radii(a: ArrayLike) -> FloatArray:
    """Magnitudes of the roots of z^2 + a1 z + a2 (poles of a normalised biquad)."""
    a_arr = np.asarray(a, dtype=np.float64)
    if a_arr.shape != (3,) or a_arr[0] != 1.0:
        raise ValueError("a must be [1, a1, a2]")
    discriminant = complex(a_arr[1] ** 2 - 4.0 * a_arr[2])
    root = discriminant**0.5
    poles = np.array([(-a_arr[1] + root) / 2.0, (-a_arr[1] - root) / 2.0])
    return np.asarray(np.abs(poles), dtype=np.float64)
