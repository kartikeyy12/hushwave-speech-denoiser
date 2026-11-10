"use strict";

// Canvas charts for the recording view. Every function draws into a 2D context in CSS pixels
// (`w` by `h`) and takes the colours from a theme object, so the same code serves the light and
// the dark page. The time axes of all charts share their left and right margins, which keeps
// them lined up when stacked.
(function (root) {
  const PAD = { l: 72, r: 64, t: 10, b: 28 };
  const PAD_FREQ = { l: 72, r: 16, t: 10, b: 32 };
  const PALETTE = [
    [0, 0, 4],
    [31, 12, 72],
    [85, 15, 109],
    [136, 34, 106],
    [186, 54, 85],
    [227, 89, 51],
    [249, 140, 10],
    [249, 201, 50],
    [252, 255, 164],
  ];
  const LUT = (() => {
    const lut = new Uint8ClampedArray(256 * 3);
    for (let i = 0; i < 256; i++) {
      const position = (i / 255) * (PALETTE.length - 1);
      const lo = Math.min(PALETTE.length - 2, Math.floor(position));
      const t = position - lo;
      for (let c = 0; c < 3; c++) lut[i * 3 + c] = PALETTE[lo][c] + (PALETTE[lo + 1][c] - PALETTE[lo][c]) * t;
    }
    return lut;
  })();

  function themeFrom(element) {
    const style = getComputedStyle(element);
    const get = (name, fallback) => style.getPropertyValue(name).trim() || fallback;
    return {
      fg: get("--fg", "#0f1722"),
      muted: get("--muted", "#5b6573"),
      line: get("--line", "#e3e6eb"),
      soft: get("--soft", "#f1f3f6"),
      surface: get("--surface", "#ffffff"),
      accent: get("--accent", "#0b6bcb"),
      original: get("--original", "#8792a2"),
      denoised: get("--denoised", "#0b6bcb"),
      band: get("--band", "rgba(11, 107, 203, 0.10)"),
      warn: get("--warn", "#b45309"),
      font: get("--sans", "system-ui, sans-serif"),
    };
  }

  function area(w, h, pad = PAD) {
    return { x0: pad.l, y0: pad.t, x1: w - pad.r, y1: h - pad.b, w: w - pad.l - pad.r, h: h - pad.t - pad.b };
  }

  function begin(ctx, w, h, dpr) {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.font = `12px system-ui, sans-serif`;
    ctx.textBaseline = "middle";
  }

  // Tick positions at 1, 2 or 5 times a power of ten, no more than about `count` of them.
  function niceTicks(min, max, count) {
    const span = max - min;
    if (!(span > 0)) return [min];
    const raw = span / count;
    const base = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 5, 10].map((m) => m * base).find((s) => s >= raw);
    const ticks = [];
    for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) ticks.push(Math.round(v / step) * step);
    return ticks;
  }

  function timeLabel(seconds, duration) {
    if (duration >= 60) return `${Math.floor(seconds / 60)}:${String(Math.round(seconds % 60)).padStart(2, "0")}`;
    return `${Number(seconds.toFixed(1))} s`;
  }

  function timeTicks(duration) {
    const steps = [0.5, 1, 2, 5, 10, 15, 30, 60, 120];
    const step = steps.find((s) => duration / s <= 9) || 120;
    const out = [];
    for (let t = 0; t <= duration + 1e-9; t += step) out.push(t);
    return out;
  }

  function axes(ctx, a, theme, opt) {
    ctx.save();
    ctx.lineWidth = 1;
    ctx.strokeStyle = theme.line;
    ctx.fillStyle = theme.muted;
    ctx.textAlign = "right";
    for (const tick of opt.yTicks) {
      const y = Math.round(a.y1 - ((tick.v - opt.yMin) / (opt.yMax - opt.yMin)) * a.h) + 0.5;
      if (opt.grid !== false) {
        ctx.beginPath();
        ctx.moveTo(a.x0, y);
        ctx.lineTo(a.x1, y);
        ctx.stroke();
      }
      ctx.fillText(tick.label, a.x0 - 8, y);
    }
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const tick of opt.xTicks) {
      const x = a.x0 + ((tick.v - opt.xMin) / (opt.xMax - opt.xMin)) * a.w;
      ctx.fillText(tick.label, Math.min(a.x1 - 6, Math.max(a.x0 + 6, x)), a.y1 + 7);
      ctx.beginPath();
      ctx.moveTo(Math.round(x) + 0.5, a.y1);
      ctx.lineTo(Math.round(x) + 0.5, a.y1 + 4);
      ctx.stroke();
    }
    ctx.strokeStyle = theme.line;
    ctx.strokeRect(a.x0 + 0.5, a.y0 + 0.5, a.w - 1, a.h - 1);
    if (opt.yTitle) {
      ctx.translate(11, a.y0 + a.h / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(opt.yTitle, 0, 0);
    }
    ctx.restore();
  }

  function dbTicks(min, max, count) {
    return niceTicks(min, max, count).map((v) => ({ v, label: String(v) }));
  }

  // ------------------------------------------------------------------ waveform
  function niceCeil(value) {
    for (const v of [0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1]) if (value <= v) return v;
    return 1;
  }

  function extremes(x, columns) {
    const lo = new Float32Array(columns);
    const hi = new Float32Array(columns);
    const n = x.length;
    for (let c = 0; c < columns; c++) {
      const from = Math.floor((c * n) / columns);
      const to = Math.max(from + 1, Math.floor(((c + 1) * n) / columns));
      let min = Infinity;
      let max = -Infinity;
      for (let i = from; i < to && i < n; i++) {
        const v = x[i];
        if (v < min) min = v;
        if (v > max) max = v;
      }
      lo[c] = min === Infinity ? 0 : min;
      hi[c] = max === -Infinity ? 0 : max;
    }
    return { lo, hi };
  }

  // d: {original, denoised (optional), fs}
  function drawWaveform(ctx, w, h, d, theme, dpr = 1) {
    begin(ctx, w, h, dpr);
    const a = area(w, h);
    const duration = d.original.length / d.fs;
    let peak = 0;
    for (const x of [d.original, d.denoised]) {
      if (!x) continue;
      for (let i = 0; i < x.length; i++) peak = Math.max(peak, Math.abs(x[i]));
    }
    const scale = niceCeil(peak * 1.02);
    const columns = Math.max(1, Math.floor(a.w));
    const draw = (x, color, alpha) => {
      const { lo, hi } = extremes(x, columns);
      ctx.globalAlpha = alpha;
      ctx.fillStyle = color;
      for (let c = 0; c < columns; c++) {
        const top = a.y0 + (0.5 - (0.5 * hi[c]) / scale) * a.h;
        const bottom = a.y0 + (0.5 - (0.5 * lo[c]) / scale) * a.h;
        ctx.fillRect(a.x0 + c, top, 1, Math.max(1, bottom - top));
      }
      ctx.globalAlpha = 1;
    };
    draw(d.original, theme.original, d.denoised ? 0.55 : 0.95);
    if (d.denoised) draw(d.denoised, theme.denoised, 0.95);
    const yTicks = [-scale, 0, scale].map((v) => ({ v, label: v === 0 ? "0" : (v > 0 ? "+" : "-") + Math.abs(v) }));
    axes(ctx, a, theme, {
      xMin: 0,
      xMax: duration,
      yMin: -scale,
      yMax: scale,
      xTicks: timeTicks(duration).map((t) => ({ v: t, label: timeLabel(t, duration) })),
      yTicks,
      yTitle: "Amplitude",
    });
  }

  // ---------------------------------------------------------------- spectrogram
  function drawColorbar(ctx, a, theme, vmin, vmax) {
    const x = a.x1 + 12;
    const width = 10;
    for (let i = 0; i < a.h; i++) {
      const lutIndex = Math.round(255 * (1 - i / Math.max(1, a.h - 1))) * 3;
      ctx.fillStyle = `rgb(${LUT[lutIndex]}, ${LUT[lutIndex + 1]}, ${LUT[lutIndex + 2]})`;
      ctx.fillRect(x, a.y0 + i, width, 1.5);
    }
    ctx.fillStyle = theme.muted;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(`${Math.round(vmax)}`, x + width + 4, a.y0 + 6);
    ctx.fillText(`${Math.round(vmin)}`, x + width + 4, a.y1 - 6);
    ctx.fillText("dB", x + width + 4, a.y0 + a.h / 2);
  }

  // d: {spec, fs, vmin, vmax}. The image is built in device pixels.
  function drawSpectrogram(ctx, w, h, d, theme, dpr = 1) {
    begin(ctx, w, h, dpr);
    const a = area(w, h);
    const spec = d.spec;
    const duration = ((spec.frames - 1) * spec.hop + spec.frame) / d.fs;
    const X0 = Math.round(a.x0 * dpr);
    const Y0 = Math.round(a.y0 * dpr);
    const PW = Math.max(1, Math.round(a.w * dpr));
    const PH = Math.max(1, Math.round(a.h * dpr));
    const bins = spec.bins;
    const mean = new Float32Array(PW * bins);
    for (let c = 0; c < PW; c++) {
      const f0 = Math.min(spec.frames - 1, Math.floor((c * spec.frames) / PW));
      const f1 = Math.max(f0 + 1, Math.min(spec.frames, Math.floor(((c + 1) * spec.frames) / PW)));
      for (let k = 0; k < bins; k++) {
        let sum = 0;
        for (let f = f0; f < f1; f++) sum += spec.db[f * bins + k];
        mean[c * bins + k] = sum / (f1 - f0);
      }
    }
    const image = ctx.createImageData(PW, PH);
    const span = d.vmax - d.vmin;
    for (let r = 0; r < PH; r++) {
      const bin = Math.min(bins - 1, Math.max(0, Math.round((1 - (r + 0.5) / PH) * (bins - 1))));
      for (let c = 0; c < PW; c++) {
        const t = Math.min(1, Math.max(0, (mean[c * bins + bin] - d.vmin) / span));
        const li = Math.round(t * 255) * 3;
        const o = (r * PW + c) * 4;
        image.data[o] = LUT[li];
        image.data[o + 1] = LUT[li + 1];
        image.data[o + 2] = LUT[li + 2];
        image.data[o + 3] = 255;
      }
    }
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.putImageData(image, X0, Y0);
    ctx.restore();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const nyquist = d.fs / 2;
    axes(ctx, a, theme, {
      xMin: 0,
      xMax: duration,
      yMin: 0,
      yMax: nyquist,
      grid: false,
      xTicks: timeTicks(duration).map((t) => ({ v: t, label: timeLabel(t, duration) })),
      yTicks: [0, 2000, 4000, 6000, 8000].filter((v) => v <= nyquist).map((v) => ({ v, label: v === 0 ? "0" : `${v / 1000} kHz` })),
      yTitle: "Frequency",
    });
    drawColorbar(ctx, a, theme, d.vmin, d.vmax);
  }

  // ----------------------------------------------------------------- line charts
  function columnSeries(values, columns, mode) {
    const out = new Float32Array(columns).fill(NaN);
    const n = values.length;
    for (let c = 0; c < columns; c++) {
      const from = Math.floor((c * n) / columns);
      const to = Math.max(from + 1, Math.floor(((c + 1) * n) / columns));
      let sum = 0;
      let count = 0;
      let best = -Infinity;
      for (let i = from; i < to && i < n; i++) {
        const v = values[i];
        if (!Number.isFinite(v)) continue;
        sum += v;
        count++;
        if (v > best) best = v;
      }
      if (count > 0) out[c] = mode === "max" ? best : sum / count;
    }
    return out;
  }

  function stroke(ctx, xs, ys, color, width, dash) {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineJoin = "round";
    ctx.setLineDash(dash || []);
    ctx.beginPath();
    let pen = false;
    for (let i = 0; i < xs.length; i++) {
      if (!Number.isFinite(ys[i])) {
        pen = false;
        continue;
      }
      if (pen) ctx.lineTo(xs[i], ys[i]);
      else ctx.moveTo(xs[i], ys[i]);
      pen = true;
    }
    ctx.stroke();
    ctx.restore();
  }

  function clip(ctx, a) {
    ctx.beginPath();
    ctx.rect(a.x0, a.y0, a.w, a.h);
    ctx.clip();
  }

  // Block energy over time. d: {original, denoised (dB per block), runs (pause runs in blocks),
  // floorDb, hop, fs}
  function drawLevels(ctx, w, h, d, theme, dpr = 1) {
    begin(ctx, w, h, dpr);
    const a = area(w, h);
    const duration = (d.original.length * d.hop) / d.fs;
    const all = [...d.original, ...d.denoised].filter((v) => Number.isFinite(v)).sort((p, q) => p - q);
    // Blocks of digital silence (the first block of a capture, for example) must not stretch the axis.
    const low = all[Math.floor(0.005 * (all.length - 1))];
    const yMax = Math.min(3, Math.ceil((all[all.length - 1] + 1) / 10) * 10);
    const yMin = Math.max(-110, Math.floor((low - 3) / 10) * 10);
    const yOf = (v) => a.y1 - ((v - yMin) / (yMax - yMin)) * a.h;
    const columns = Math.max(1, Math.floor(a.w));
    ctx.save();
    clip(ctx, a);
    ctx.fillStyle = theme.band;
    for (const [from, to] of d.runs) {
      const x0 = a.x0 + (from / d.original.length) * a.w;
      const x1 = a.x0 + (to / d.original.length) * a.w;
      ctx.fillRect(x0, a.y0, Math.max(1, x1 - x0), a.h);
    }
    ctx.restore();
    axes(ctx, a, theme, {
      xMin: 0,
      xMax: duration,
      yMin,
      yMax,
      xTicks: timeTicks(duration).map((t) => ({ v: t, label: timeLabel(t, duration) })),
      yTicks: dbTicks(yMin, yMax, 6),
      yTitle: "Block energy, dB",
    });
    ctx.save();
    clip(ctx, a);
    const xs = Float32Array.from({ length: columns }, (_, c) => a.x0 + c + 0.5);
    const toY = (series) => Float32Array.from(series, (v) => (Number.isFinite(v) ? yOf(v) : NaN));
    stroke(ctx, xs, toY(columnSeries(d.original, columns, "mean")), theme.original, 1.25);
    stroke(ctx, xs, toY(columnSeries(d.denoised, columns, "mean")), theme.denoised, 1.5);
    if (Number.isFinite(d.floorDb)) {
      const y = yOf(d.floorDb + 1);
      stroke(ctx, [a.x0, a.x1], [y, y], theme.warn, 1, [5, 4]);
    }
    ctx.restore();
  }

  // Average spectra. d: {series: [{db, color, dash}], fs}
  function drawSpectra(ctx, w, h, d, theme, dpr = 1) {
    begin(ctx, w, h, dpr);
    const a = area(w, h, PAD_FREQ);
    const nyquist = d.fs / 2;
    const all = d.series.flatMap((s) => Array.from(s.db));
    const yMax = Math.ceil((Math.max(...all) + 1) / 10) * 10;
    const yMin = Math.max(Math.floor((Math.min(...all) - 1) / 10) * 10, yMax - 80);
    axes(ctx, a, theme, {
      xMin: 0,
      xMax: nyquist,
      yMin,
      yMax,
      xTicks: (a.w < 420 ? [0, 2000, 4000, 6000, 8000] : [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000])
        .filter((v) => v <= nyquist)
        .map((v) => ({ v, label: v === 0 ? "0" : `${v / 1000} kHz` })),
      yTicks: dbTicks(yMin, yMax, 6),
      yTitle: "Power per bin, dB",
    });
    ctx.save();
    clip(ctx, a);
    for (const s of d.series) {
      const bins = s.db.length;
      const xs = Float32Array.from({ length: bins }, (_, k) => a.x0 + (k / (bins - 1)) * a.w);
      const ys = Float32Array.from(s.db, (v) => a.y1 - ((v - yMin) / (yMax - yMin)) * a.h);
      stroke(ctx, xs, ys, theme[s.color] || s.color, s.dash ? 1.4 : 1.8, s.dash);
    }
    ctx.restore();
  }

  // Processing time per block. d: {compute, rtt (ms per block), blockMs}
  function drawTiming(ctx, w, h, d, theme, dpr = 1) {
    begin(ctx, w, h, dpr);
    const a = area(w, h);
    const duration = (d.compute.length * d.blockMs) / 1000;
    const finite = [...d.compute, ...d.rtt].filter((v) => Number.isFinite(v));
    const top = Math.max(d.blockMs * 1.25, finite.length ? Math.max(...finite) * 1.1 : 0);
    const tickList = niceTicks(0, top, 5);
    const tickStep = tickList.length > 1 ? tickList[1] - tickList[0] : top;
    const yMax = Math.ceil(top / tickStep - 1e-9) * tickStep;
    const yOf = (v) => a.y1 - (v / yMax) * a.h;
    const columns = Math.max(1, Math.floor(a.w));
    axes(ctx, a, theme, {
      xMin: 0,
      xMax: duration,
      yMin: 0,
      yMax,
      xTicks: timeTicks(duration).map((t) => ({ v: t, label: timeLabel(t, duration) })),
      yTicks: niceTicks(0, yMax, 5).map((v) => ({ v, label: String(v) })),
      yTitle: "Milliseconds",
    });
    ctx.save();
    clip(ctx, a);
    const xs = Float32Array.from({ length: columns }, (_, c) => a.x0 + c + 0.5);
    const toY = (series) => Float32Array.from(series, (v) => (Number.isFinite(v) ? yOf(v) : NaN));
    stroke(ctx, xs, toY(columnSeries(d.rtt, columns, "max")), theme.original, 1.25);
    stroke(ctx, xs, toY(columnSeries(d.compute, columns, "max")), theme.denoised, 1.5);
    const budget = yOf(d.blockMs);
    stroke(ctx, [a.x0, a.x1], [budget, budget], theme.warn, 1, [5, 4]);
    ctx.restore();
  }

  // Level while recording. d: {peaks (peak per block), spanBlocks}
  function drawEnvelope(ctx, w, h, d, theme, dpr = 1) {
    begin(ctx, w, h, dpr);
    const mid = h / 2;
    ctx.fillStyle = theme.line;
    ctx.fillRect(0, Math.round(mid), w, 1);
    const columns = Math.max(1, Math.floor(w));
    const per = d.spanBlocks / columns;
    ctx.fillStyle = theme.denoised;
    for (let c = 0; c < columns; c++) {
      const from = Math.floor(c * per);
      if (from >= d.peaks.length) break;
      const to = Math.max(from + 1, Math.floor((c + 1) * per));
      let peak = 0;
      for (let i = from; i < to && i < d.peaks.length; i++) peak = Math.max(peak, d.peaks[i]);
      const half = Math.max(0.5, Math.min(1, peak * 2) * (h / 2 - 2));
      ctx.fillRect(c, mid - half, 1, 2 * half);
    }
  }

  // Position of a pointer inside a time chart, as a fraction of the recording.
  function timeFraction(x, width) {
    return Math.min(1, Math.max(0, (x - PAD.l) / (width - PAD.l - PAD.r)));
  }

  const api = {
    PAD,
    PAD_FREQ,
    themeFrom,
    niceTicks,
    timeTicks,
    timeLabel,
    drawWaveform,
    drawSpectrogram,
    drawLevels,
    drawSpectra,
    drawTiming,
    drawEnvelope,
    timeFraction,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.HushwaveCharts = api;
})(typeof self !== "undefined" ? self : this);
