"use strict";

// Signal analysis for the recording view: spectrogram, block levels, pause and speech
// segmentation, the measurements of a recording that has no clean reference, whole-recording
// processing through an engine, and WAV encoding. Pure functions, no DOM, so Node can test them
// against the Python implementation.
(function (root) {
  const FRAME = 512;
  const HOP = 256;
  const BINS = FRAME / 2 + 1;
  const TINY = 1e-20;
  const FLOOR_PERCENTILE = 10;
  const FLOOR_OFFSET_DB = 6;
  const PAUSE_MARGIN_DB = 1;
  const SPEECH_MARGIN_DB = 10;

  // ---------------------------------------------------------------- spectrogram
  function fft(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        const tr = re[i];
        re[i] = re[j];
        re[j] = tr;
        const ti = im[i];
        im[i] = im[j];
        im[j] = ti;
      }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const angle = (-2 * Math.PI) / len;
      const wr = Math.cos(angle);
      const wi = Math.sin(angle);
      for (let start = 0; start < n; start += len) {
        let cr = 1;
        let ci = 0;
        for (let k = 0; k < len / 2; k++) {
          const a = start + k;
          const b = a + len / 2;
          const xr = re[b] * cr - im[b] * ci;
          const xi = re[b] * ci + im[b] * cr;
          re[b] = re[a] - xr;
          im[b] = im[a] - xi;
          re[a] += xr;
          im[a] += xi;
          const next = cr * wr - ci * wi;
          ci = cr * wi + ci * wr;
          cr = next;
        }
      }
    }
  }

  const WINDOW = (() => {
    const w = new Float64Array(FRAME);
    for (let i = 0; i < FRAME; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / FRAME);
    return w;
  })();
  const WINDOW_POWER = WINDOW.reduce((sum, v) => sum + v * v, 0);

  // Power per bin (relative to full scale, normalised by the window energy) and its dB value.
  function spectrogram(x) {
    const frames = x.length >= FRAME ? Math.floor((x.length - FRAME) / HOP) + 1 : 0;
    const power = new Float32Array(frames * BINS);
    const db = new Float32Array(frames * BINS);
    const re = new Float64Array(FRAME);
    const im = new Float64Array(FRAME);
    for (let f = 0; f < frames; f++) {
      const offset = f * HOP;
      for (let i = 0; i < FRAME; i++) {
        re[i] = x[offset + i] * WINDOW[i];
        im[i] = 0;
      }
      fft(re, im);
      for (let k = 0; k < BINS; k++) {
        const p = (re[k] * re[k] + im[k] * im[k]) / WINDOW_POWER;
        power[f * BINS + k] = p;
        db[f * BINS + k] = 10 * Math.log10(p + 1e-12);
      }
    }
    return { frames, bins: BINS, hop: HOP, frame: FRAME, power, db };
  }

  // Mean power over the selected frames, per bin, in dB. Null when no frame is selected.
  function averageSpectrumDb(spec, frameMask) {
    const sum = new Float64Array(BINS);
    let count = 0;
    for (let f = 0; f < spec.frames; f++) {
      if (frameMask && !frameMask[f]) continue;
      for (let k = 0; k < BINS; k++) sum[k] += spec.power[f * BINS + k];
      count++;
    }
    if (count === 0) return null;
    const out = new Float32Array(BINS);
    for (let k = 0; k < BINS; k++) out[k] = 10 * Math.log10(sum[k] / count + 1e-12);
    return out;
  }

  function percentile(values, q) {
    const sorted = Float64Array.from(values).sort();
    if (sorted.length === 0) return NaN;
    const position = ((sorted.length - 1) * q) / 100;
    const lo = Math.floor(position);
    const hi = Math.min(sorted.length - 1, lo + 1);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (position - lo);
  }

  // dB value below which the given share of spectrogram cells lies (sampled, so it stays fast).
  function dbPercentile(spec, q) {
    const step = Math.max(1, Math.floor(spec.db.length / 50000));
    const sample = [];
    for (let i = 0; i < spec.db.length; i += step) sample.push(spec.db[i]);
    return percentile(sample, q);
  }

  // ------------------------------------------------------------ levels and pauses
  // Mean-square energy of each complete block in dB (a full-scale sine is -3 dB).
  function blockEnergyDb(x, block = HOP) {
    const count = Math.floor(x.length / block);
    const out = new Float64Array(count);
    for (let b = 0; b < count; b++) {
      let sum = 0;
      for (let i = b * block; i < (b + 1) * block; i++) sum += x[i] * x[i];
      out[b] = 10 * Math.log10(sum / block + TINY);
    }
    return out;
  }

  // The same rule as the Python evaluation of recordings without a clean reference: the floor
  // is the 10th percentile of the block energies plus 6 dB. Pauses lie within 1 dB above the
  // floor, speech more than 10 dB above it. The partial last block counts as neither.
  function segment(x) {
    const energy = blockEnergyDb(x);
    const blocks = Math.ceil(x.length / HOP);
    const pause = new Uint8Array(blocks);
    const speech = new Uint8Array(blocks);
    const floorDb = percentile(energy, FLOOR_PERCENTILE) + FLOOR_OFFSET_DB;
    for (let b = 0; b < energy.length; b++) {
      pause[b] = energy[b] <= floorDb + PAUSE_MARGIN_DB ? 1 : 0;
      speech[b] = energy[b] > floorDb + SPEECH_MARGIN_DB ? 1 : 0;
    }
    return { energy, pause, speech, floorDb, blocks };
  }

  // Frame k of the spectrogram covers blocks k and k + 1.
  function frameMask(blockFlags, frames, mode) {
    const mask = new Uint8Array(frames);
    for (let f = 0; f < frames; f++) {
      const a = blockFlags[f];
      const b = blockFlags[f + 1];
      mask[f] = mode === "all" ? (a && b ? 1 : 0) : a || b ? 1 : 0;
    }
    return mask;
  }

  function runs(flags) {
    const out = [];
    let start = -1;
    for (let i = 0; i <= flags.length; i++) {
      const on = i < flags.length && flags[i] === 1;
      if (on && start < 0) start = i;
      else if (!on && start >= 0) {
        out.push([start, i]);
        start = -1;
      }
    }
    return out;
  }

  const toDb = (value) => 10 * Math.log10(Math.max(value, TINY));

  function peakDb(x) {
    let peak = 0;
    for (let i = 0; i < x.length; i++) peak = Math.max(peak, Math.abs(x[i]));
    return peak <= 0 ? -Infinity : 20 * Math.log10(peak);
  }

  function rmsDb(x) {
    let sum = 0;
    for (let i = 0; i < x.length; i++) sum += x[i] * x[i];
    return toDb(x.length ? sum / x.length : 0);
  }

  // Measurements of one processed recording. `noisy` and `enhanced` have the same length.
  function measure(noisy, enhanced, segmentation) {
    const seg = segmentation || segment(noisy);
    const n = noisy.length;
    let beforePause = 0;
    let afterPause = 0;
    let beforeSpeech = 0;
    let afterSpeech = 0;
    let pauseSamples = 0;
    let speechSamples = 0;
    for (let b = 0; b < seg.energy.length; b++) {
      const flagPause = seg.pause[b] === 1;
      const flagSpeech = seg.speech[b] === 1;
      if (!flagPause && !flagSpeech) continue;
      for (let i = b * HOP; i < (b + 1) * HOP && i < n; i++) {
        const y2 = noisy[i] * noisy[i];
        const z2 = enhanced[i] * enhanced[i];
        if (flagPause) {
          beforePause += y2;
          afterPause += z2;
          pauseSamples++;
        } else {
          beforeSpeech += y2;
          afterSpeech += z2;
          speechSamples++;
        }
      }
    }
    const pauseBlocks = seg.pause.reduce((a, b) => a + b, 0);
    const speechBlocks = seg.speech.reduce((a, b) => a + b, 0);
    return {
      durationS: n / 16000,
      floorDb: seg.floorDb,
      pauseBlocks,
      speechBlocks,
      totalBlocks: seg.blocks,
      noiseReductionPauseDb:
        pauseSamples > 0 && beforePause > 0 ? 10 * Math.log10(beforePause / Math.max(afterPause, TINY)) : NaN,
      residualPauseDbfs: pauseSamples > 0 ? toDb(afterPause / pauseSamples) : NaN,
      originalPauseDbfs: pauseSamples > 0 ? toDb(beforePause / pauseSamples) : NaN,
      speechLevelChangeDb:
        speechSamples > 0 && beforeSpeech > 0 ? toDb(afterSpeech / speechSamples) - toDb(beforeSpeech / speechSamples) : NaN,
      peakInDbfs: peakDb(noisy),
      peakOutDbfs: peakDb(enhanced),
      rmsInDbfs: rmsDb(noisy),
      rmsOutDbfs: rmsDb(enhanced),
    };
  }

  function describeValues(values) {
    const clean = Array.from(values).filter((v) => Number.isFinite(v));
    if (clean.length === 0) return { mean: NaN, p95: NaN, max: NaN };
    const sum = clean.reduce((a, b) => a + b, 0);
    return { mean: sum / clean.length, p95: percentile(clean, 95), max: Math.max(...clean) };
  }

  // ------------------------------------------------------------ whole recording
  // Streams a whole recording through an engine and returns the output lined up with the input.
  //   engine  open engine with submit(block, token), stats and onOutput (see engines.js)
  //   info    {fs, block, delay} as returned by engine.open()
  // The engine delays its output by `delay` samples, so enough silence is fed after the end to
  // bring out the last samples, and the first `delay` output samples are dropped.
  function denoiseWhole(engine, samples, info, options = {}) {
    const block = info.block;
    const delay = info.delay;
    const n = samples.length;
    const total = Math.ceil((n + delay) / block);
    const windowSize = options.window || 12;
    const stallMs = options.stallMs || 5000;
    const intervalMs = options.intervalMs === undefined ? 10 : options.intervalMs;
    const now = options.now || (() => Date.now());
    const stream = new Float32Array(total * block);
    const computeMs = new Float32Array(total).fill(NaN);
    const rttMs = new Float32Array(total).fill(NaN);
    const got = new Uint8Array(total);
    const lostAtStart = engine.stats.lost;
    const droppedAtStart = engine.stats.dropped;
    engine.maxQueue = Math.max(engine.maxQueue || 0, windowSize + 8);
    const previous = engine.onOutput;
    return new Promise((resolve, reject) => {
      let sent = 0;
      let received = 0;
      let lastDone = 0;
      let lastChange = now();
      let timer = null;
      const missing = () => engine.stats.lost - lostAtStart + (engine.stats.dropped - droppedAtStart);
      const finish = (error) => {
        clearInterval(timer);
        engine.onOutput = previous;
        if (error) {
          reject(error);
          return;
        }
        resolve({
          output: stream.slice(delay, delay + n),
          computeMs,
          rttMs,
          blocks: total,
          received,
          lost: engine.stats.lost - lostAtStart,
          dropped: engine.stats.dropped - droppedAtStart,
        });
      };
      engine.onOutput = (out, meta, token) => {
        if (!(token >= 0 && token < total) || got[token]) return;
        got[token] = 1;
        stream.set(out, token * block);
        computeMs[token] = meta.computeMs;
        rttMs[token] = meta.rttMs;
        received++;
      };
      const step = () => {
        const problem = options.failed ? options.failed() : null;
        if (problem) {
          finish(problem);
          return;
        }
        if (options.isCancelled && options.isCancelled()) {
          const error = new Error("Cancelled");
          error.name = "AbortError";
          finish(error);
          return;
        }
        const done = received + missing();
        if (done !== lastDone) {
          lastDone = done;
          lastChange = now();
        }
        if (options.onProgress) options.onProgress(Math.min(done, total), total);
        if (done >= total) {
          finish(null);
          return;
        }
        if (now() - lastChange > stallMs) {
          finish(new Error("The engine stopped answering."));
          return;
        }
        while (sent < total && sent - done < windowSize) {
          const piece = new Float32Array(block);
          piece.set(samples.subarray(Math.min(n, sent * block), Math.min(n, (sent + 1) * block)));
          engine.submit(piece, sent);
          sent++;
        }
      };
      timer = setInterval(step, intervalMs);
      step();
    });
  }

  // ------------------------------------------------------------------------- WAV
  function wavBytes(samples, fs) {
    const buffer = new ArrayBuffer(44 + 2 * samples.length);
    const view = new DataView(buffer);
    const text = (offset, value) => [...value].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)));
    text(0, "RIFF");
    view.setUint32(4, 36 + 2 * samples.length, true);
    text(8, "WAVEfmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, fs, true);
    view.setUint32(28, 2 * fs, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    text(36, "data");
    view.setUint32(40, 2 * samples.length, true);
    for (let i = 0; i < samples.length; i++) {
      view.setInt16(44 + 2 * i, Math.max(-32768, Math.min(32767, Math.round(samples[i] * 32768))), true);
    }
    return buffer;
  }

  const api = {
    FRAME,
    HOP,
    BINS,
    fft,
    spectrogram,
    averageSpectrumDb,
    dbPercentile,
    percentile,
    blockEnergyDb,
    segment,
    frameMask,
    runs,
    measure,
    describeValues,
    peakDb,
    rmsDb,
    denoiseWhole,
    wavBytes,
    wavBlob: (samples, fs) => new Blob([wavBytes(samples, fs)], { type: "audio/wav" }),
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.HushwaveAnalysis = api;
})(typeof self !== "undefined" ? self : this);
