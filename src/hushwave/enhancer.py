"""Batch and streaming speech enhancer (docs/THEORY.md section 4.5).

Signal path, identical in both forms:

    high-pass biquad -> sqrt-Hann STFT -> |Y|^2 -> noise PSD tracker -> gain rule -> ISTFT

``enhance`` runs it on a whole signal with the vectorised STFT of ``stft.py``. ``StreamingEnhancer``
runs it block by block with state carried across calls, and reproduces ``enhance`` to rounding error
for any sequence of block sizes.

Layout and delay. The stream is primed with N - H zeros, as in ``stft.py``. Frame f covers padded
samples [f H, f H + N) and is complete once input sample (f + 1) H - 1 has arrived. After frame f,
padded samples [f H, f H + H) are final. Output sample m therefore becomes final between H and
N - 1 samples after input sample m arrived. A block-in/block-out engine needs one fixed delay, so
the engine delays its output by the worst case, D = N - 1 samples: ``process`` returns exactly as
many samples as it was given, the first D samples of the stream are zeros, and ``flush`` returns the
last D samples.

The rule and the noise tracker defaults below are provisional: M8 chooses them on DEV only.
"""

from __future__ import annotations

import platform
import time
import tracemalloc
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hushwave.baselines import _as_signal, spectral_denoise
from hushwave.config import DEFAULT, Config
from hushwave.fft import irfft, rfft
from hushwave.filters import Biquad, highpass_biquad
from hushwave.gains import GainStage
from hushwave.noise_est import make_estimator
from hushwave.stft import istft, num_frames, stft
from hushwave.windows import sqrt_hann

FloatArray = NDArray[np.float64]
DEFAULT_RULE = "lsa"
DEFAULT_ESTIMATOR = "mcra"


def enhance(
    x: ArrayLike,
    cfg: Config = DEFAULT,
    rule: str = DEFAULT_RULE,
    estimator: str = DEFAULT_ESTIMATOR,
    prefilter: bool = True,
    gain_overrides: dict[str, float] | None = None,
    estimator_overrides: dict[str, float] | None = None,
) -> FloatArray:
    """Enhance a whole mono signal. Returns exactly len(x) samples with zero delay."""
    signal = _as_signal(x)
    if prefilter:
        b, a = highpass_biquad(cfg.hp_cutoff_hz, cfg.fs)
        signal = Biquad(b, a).process(signal)
    return spectral_denoise(signal, rule, estimator, cfg, gain_overrides, estimator_overrides)


@dataclass(frozen=True)
class Components:
    """The system output split into the parts produced from the clean speech and from the noise."""

    speech: FloatArray
    noise: FloatArray
    output: FloatArray


def gain_components(
    clean: ArrayLike,
    noise: ArrayLike,
    cfg: Config = DEFAULT,
    rule: str = DEFAULT_RULE,
    estimator: str = DEFAULT_ESTIMATOR,
    prefilter: bool = True,
    gain_overrides: dict[str, float] | None = None,
    estimator_overrides: dict[str, float] | None = None,
) -> Components:
    """Run the system on clean + noise, then apply the very same gains to each part separately.

    The gains are computed from the mixture exactly as `enhance` does, and the STFT, the
    high-pass filter and the gain multiplication are linear, so speech + noise equals the output
    up to rounding. The speech part is what the system did to the speech, the noise part is the
    residual noise. Only possible for systems that work by applying gains.
    """
    x = _as_signal(clean)
    d = _as_signal(noise)
    if x.shape != d.shape:
        raise ValueError("clean and noise must have the same length")
    parts = [x, d, x + d]
    if prefilter:
        b, a = highpass_biquad(cfg.hp_cutoff_hz, cfg.fs)
        parts = [Biquad(b, a).process(p) for p in parts]
    spec_x, spec_d, spec_y = (stft(p, cfg.frame_len, cfg.hop) for p in parts)
    power = np.abs(spec_y) ** 2
    tracker = make_estimator(estimator, cfg.n_bins, cfg, **(estimator_overrides or {}))
    stage = GainStage(rule, cfg.n_bins, cfg, **(gain_overrides or {}))
    gains = np.empty_like(power)
    for index, frame in enumerate(power):
        gains[index] = stage.process(frame, tracker.update(frame))

    def synth(spectrum: NDArray[np.complex128]) -> FloatArray:
        return istft(spectrum * gains, x.size, cfg.frame_len, cfg.hop)

    return Components(speech=synth(spec_x), noise=synth(spec_d), output=synth(spec_y))


class StreamingEnhancer:
    """Block-by-block enhancer with a fixed delay of ``delay_samples`` = frame_len - 1."""

    def __init__(
        self,
        cfg: Config = DEFAULT,
        rule: str = DEFAULT_RULE,
        estimator: str = DEFAULT_ESTIMATOR,
        prefilter: bool = True,
        gain_overrides: dict[str, float] | None = None,
        estimator_overrides: dict[str, float] | None = None,
    ) -> None:
        self.cfg = cfg
        self.frame_len = cfg.frame_len
        self.hop = cfg.hop
        self.delay_samples = cfg.latency_max_samples
        self._lead = cfg.frame_len - cfg.hop
        self._window = sqrt_hann(cfg.frame_len)
        self._biquad: Biquad | None = None
        if prefilter:
            b, a = highpass_biquad(cfg.hp_cutoff_hz, cfg.fs)
            self._biquad = Biquad(b, a)
        self._tracker = make_estimator(estimator, cfg.n_bins, cfg, **(estimator_overrides or {}))
        self._stage = GainStage(rule, cfg.n_bins, cfg, **(gain_overrides or {}))
        self.reset()

    @property
    def delay_ms(self) -> float:
        return 1000.0 * self.delay_samples / self.cfg.fs

    def reset(self) -> None:
        """Forget everything and start a new stream."""
        if self._biquad is not None:
            self._biquad.reset()
        self._tracker.reset()
        self._stage.reset()
        self._pending: FloatArray = np.zeros(self._lead)
        self._overlap: FloatArray = np.zeros(self.hop)
        self._queue: FloatArray = np.zeros(self.delay_samples)
        self._frames_done = 0
        self._samples_in = 0
        self._samples_out = 0

    def state_bytes(self) -> int:
        """Bytes held in the engine's own buffers right now (excludes estimator internals)."""
        return int(self._pending.nbytes + self._overlap.nbytes + self._queue.nbytes)

    def _run_frames(self) -> None:
        pos = 0
        pending = self._pending
        while pending.size - pos >= self.frame_len:
            frame = pending[pos : pos + self.frame_len]
            spectrum = rfft(frame * self._window)
            power = np.abs(spectrum) ** 2
            noise = self._tracker.update(power)
            gain = self._stage.process(power, noise)
            segment = irfft(spectrum * gain, n=self.frame_len) * self._window
            final = segment[: self.hop] + self._overlap
            self._overlap = segment[self.hop :].copy()
            if self._frames_done >= 1:
                self._queue = np.concatenate((self._queue, final))
            self._frames_done += 1
            pos += self.hop
        self._pending = pending[pos:].copy()

    def process(self, block: ArrayLike) -> FloatArray:
        """Feed a block of any length (0, 1, larger than a frame); get as many samples back."""
        data = np.asarray(block, dtype=np.float64)
        if data.ndim != 1:
            raise ValueError(f"expected a 1-D block, got shape {data.shape}")
        if not np.all(np.isfinite(data)):
            raise ValueError("block contains NaN or Inf")
        if data.size == 0:
            return np.zeros(0)
        filtered = data if self._biquad is None else self._biquad.process(data)
        self._pending = np.concatenate((self._pending, filtered))
        self._samples_in += data.size
        self._run_frames()
        out = self._queue[: data.size].copy()
        self._queue = self._queue[data.size :]
        self._samples_out += data.size
        return out

    def flush(self) -> FloatArray:
        """End the stream: return the last delay_samples samples, then reset for a new stream."""
        total = self._samples_in
        if total == 0:
            self.reset()
            return np.zeros(0)
        frames = num_frames(total, self.frame_len, self.hop)
        tail = (frames - 1) * self.hop + self.frame_len - self._lead - total
        self._pending = np.concatenate((self._pending, np.zeros(tail)))
        self._run_frames()
        keep = self.delay_samples + total - self._samples_out
        out = self._queue[:keep].copy()
        self.reset()
        return out


def stream(
    x: ArrayLike,
    block_sizes: Sequence[int],
    cfg: Config = DEFAULT,
    rule: str = DEFAULT_RULE,
    estimator: str = DEFAULT_ESTIMATOR,
    prefilter: bool = True,
    gain_overrides: dict[str, float] | None = None,
    estimator_overrides: dict[str, float] | None = None,
) -> FloatArray:
    """Run `x` through a StreamingEnhancer using the given block sizes (cycled), aligned to x.

    Returns the output with the fixed delay removed, so it is directly comparable to enhance(x).
    """
    signal = _as_signal(x)
    if len(block_sizes) == 0 or any(size < 1 for size in block_sizes):
        raise ValueError("block_sizes must be a non-empty sequence of positive integers")
    engine = StreamingEnhancer(cfg, rule, estimator, prefilter, gain_overrides, estimator_overrides)
    pieces: list[FloatArray] = []
    pos = 0
    index = 0
    while pos < signal.size:
        size = block_sizes[index % len(block_sizes)]
        pieces.append(engine.process(signal[pos : pos + size]))
        pos += size
        index += 1
    pieces.append(engine.flush())
    return np.concatenate(pieces)[engine.delay_samples :]


# --------------------------------------------------------------------------- measurements
@dataclass(frozen=True)
class DelayMeasurement:
    expected_samples: int
    measured_samples: int


def measure_delay(cfg: Config = DEFAULT, block: int = 160) -> DelayMeasurement:
    """Measure the stream delay with an impulse through unit gains and no pre-filter.

    With g_min = 1 every gain is exactly 1, so the engine is a pure delay and the impulse must
    reappear at index delay_samples.
    """
    engine = StreamingEnhancer(cfg, prefilter=False, gain_overrides={"g_min": 1.0})
    length = 4 * cfg.frame_len
    probe = np.zeros(length)
    probe[0] = 1.0
    out = [engine.process(probe[i : i + block]) for i in range(0, length, block)]
    stream_out = np.concatenate(out)
    return DelayMeasurement(engine.delay_samples, int(np.argmax(np.abs(stream_out))))


@dataclass(frozen=True)
class BlockTiming:
    block: int
    seconds_per_audio_second: float
    worst_block_ms: float
    block_budget_ms: float


def measure_timing(
    cfg: Config = DEFAULT,
    seconds: float = 6.0,
    blocks: Sequence[int] = (128, 256, 480, 1024),
    seed: int = 20260104,
) -> list[BlockTiming]:
    """Real-time factor and worst-case block time of the streaming engine (machine dependent)."""
    rng = np.random.default_rng(seed)
    signal = 0.05 * rng.standard_normal(round(seconds * cfg.fs))
    results = []
    for size in blocks:
        engine = StreamingEnhancer(cfg)
        worst = 0.0
        start = time.perf_counter()
        for pos in range(0, signal.size, size):
            t0 = time.perf_counter()
            engine.process(signal[pos : pos + size])
            worst = max(worst, time.perf_counter() - t0)
        total = time.perf_counter() - start
        results.append(
            BlockTiming(
                block=size,
                seconds_per_audio_second=total / seconds,
                worst_block_ms=1000.0 * worst,
                block_budget_ms=1000.0 * size / cfg.fs,
            )
        )
    return results


def measure_memory(cfg: Config = DEFAULT, seconds: float = 10.0, block: int = 256) -> int:
    """Peak Python-heap bytes allocated while streaming `seconds` of audio (tracemalloc)."""
    rng = np.random.default_rng(1)
    signal = 0.05 * rng.standard_normal(round(seconds * cfg.fs))
    engine = StreamingEnhancer(cfg)
    tracemalloc.start()
    try:
        for pos in range(0, signal.size, block):
            engine.process(signal[pos : pos + block])
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return int(peak)


def performance_report(cfg: Config = DEFAULT, seconds: float = 6.0) -> dict[str, Any]:
    """Everything M6 reports about the engine, as plain JSON-ready values (machine dependent)."""
    delay = measure_delay(cfg)
    timings = measure_timing(cfg, seconds)
    rng = np.random.default_rng(20260105)
    probe = 0.1 * rng.standard_normal(round(3 * cfg.fs))
    sizes = [1, cfg.hop, 777, cfg.frame_len + 1, 53]
    difference = float(np.max(np.abs(stream(probe, sizes, cfg) - enhance(probe, cfg))))
    engine = StreamingEnhancer(cfg)
    return {
        "platform": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "machine": platform.machine(),
            "system": platform.system(),
        },
        "algorithmic_delay_samples": delay.expected_samples,
        "algorithmic_delay_ms": 1000.0 * delay.expected_samples / cfg.fs,
        "measured_delay_samples": delay.measured_samples,
        "per_sample_delay_range_samples": [cfg.latency_min_samples, cfg.latency_max_samples],
        "streaming_vs_batch_max_abs_difference": difference,
        "timing": [
            {
                "block": t.block,
                "real_time_factor": t.seconds_per_audio_second,
                "worst_block_ms": t.worst_block_ms,
                "block_budget_ms": t.block_budget_ms,
            }
            for t in timings
        ],
        "peak_heap_bytes_10s_stream": measure_memory(cfg),
        "engine_buffer_bytes_after_construction": engine.state_bytes(),
    }
