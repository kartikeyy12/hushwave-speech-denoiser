"use strict";
// The chart code against a context that only records what is drawn: no errors, something is
// drawn, and the helper functions give the expected values.
const assert = require("node:assert/strict");
const A = require("../../demo/web/analysis.js");
const C = require("../../demo/web/charts.js");

function fakeContext() {
  const calls = {};
  const target = {
    createImageData: (w, h) => {
      calls.createImageData = [w, h];
      return { data: new Uint8ClampedArray(w * h * 4), width: w, height: h };
    },
    putImageData: (image, x, y) => {
      calls.putImageData = [image.width, image.height, x, y];
    },
  };
  return new Proxy(target, {
    get(t, name) {
      if (name === "calls") return calls;
      if (name in t) return t[name];
      return () => {
        calls[name] = (calls[name] || 0) + 1;
      };
    },
    set(t, name, value) {
      t[name] = value;
      return true;
    },
  });
}

const theme = {
  fg: "#000", muted: "#555", line: "#ddd", soft: "#eee", surface: "#fff", accent: "#06c",
  original: "#888", denoised: "#06c", band: "rgba(0,0,255,0.1)", warn: "#b45309", font: "sans-serif",
};
const fs = 16000;
const x = Float32Array.from({ length: fs * 3 }, (_, i) => 0.3 * Math.sin(i / 20) * (i > fs ? 1 : 0.05));
const z = x.map((v) => v * 0.5);
const spec = A.spectrogram(x);
const seg = A.segment(x);
const W = 600;

let ctx = fakeContext();
C.drawWaveform(ctx, W, 200, { original: x, denoised: z, fs }, theme, 2);
assert.ok(ctx.calls.fillRect > 400, "waveform columns");
ctx = fakeContext();
C.drawWaveform(ctx, W, 200, { original: x, denoised: null, fs }, theme, 1);
assert.ok(ctx.calls.fillRect > 400);

ctx = fakeContext();
C.drawSpectrogram(ctx, W, 220, { spec, fs, vmin: -90, vmax: -10 }, theme, 2);
const area = { w: W - C.PAD.l - C.PAD.r, h: 220 - C.PAD.t - C.PAD.b };
assert.deepEqual(ctx.calls.putImageData.slice(0, 2), [Math.round(area.w * 2), Math.round(area.h * 2)]);

ctx = fakeContext();
C.drawLevels(ctx, W, 220, { original: seg.energy, denoised: A.blockEnergyDb(z), runs: A.runs(seg.pause), floorDb: seg.floorDb, hop: 256, fs }, theme, 1);
assert.ok(ctx.calls.stroke > 3 && ctx.calls.fillText > 4);

ctx = fakeContext();
const series = [
  { db: A.averageSpectrumDb(spec), color: "original" },
  { db: A.averageSpectrumDb(A.spectrogram(z)), color: "denoised", dash: [5, 4] },
];
C.drawSpectra(ctx, 300, 260, { series, fs }, theme, 1);
assert.ok(ctx.calls.stroke >= 2);

ctx = fakeContext();
const blocks = 200;
C.drawTiming(ctx, W, 220, { compute: new Float32Array(blocks).fill(0.5), rtt: new Float32Array(blocks).fill(4), blockMs: 16 }, theme, 1);
assert.ok(ctx.calls.stroke > 3);

ctx = fakeContext();
C.drawEnvelope(ctx, W, 84, { peaks: Array.from({ length: 300 }, () => 0.2), spanBlocks: 625 }, theme, 1);
assert.ok(ctx.calls.fillRect > 100);

// helpers
assert.deepEqual(C.niceTicks(0, 14, 7), [0, 2, 4, 6, 8, 10, 12, 14]);
assert.deepEqual(C.niceTicks(-70, -10, 6), [-70, -60, -50, -40, -30, -20, -10]);
assert.deepEqual(C.timeTicks(14), [0, 2, 4, 6, 8, 10, 12, 14]);
assert.equal(C.timeLabel(5, 14), "5 s");
assert.equal(C.timeLabel(65, 120), "1:05");
assert.equal(C.timeFraction(C.PAD.l, W), 0);
assert.equal(C.timeFraction(W - C.PAD.r, W), 1);
assert.equal(C.timeFraction(-50, W), 0);
assert.equal(C.timeFraction(10000, W), 1);
console.log("charts: ok");
