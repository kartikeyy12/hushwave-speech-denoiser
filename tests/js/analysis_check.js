"use strict";
// Runs the recording analysis on two raw float32 files and writes the numbers as JSON, so that
// Python can compare them with its own implementation.
// Usage: node analysis_check.js <noisy.f32> <enhanced.f32> <out.json>
const fs = require("node:fs");
const A = require("../../demo/web/analysis.js");

const read = (path) => {
  const raw = fs.readFileSync(path);
  return new Float32Array(raw.buffer, raw.byteOffset, raw.length / 4);
};
const [noisyPath, enhancedPath, outPath] = process.argv.slice(2);
const y = read(noisyPath);
const z = read(enhancedPath);
const seg = A.segment(y);
const metrics = A.measure(y, z, seg);
const spec = A.spectrogram(y);
const picks = [0, 7, spec.frames - 1];
const frames = picks.map((f) => Array.from(spec.db.subarray(f * spec.bins, (f + 1) * spec.bins)));
const mask = A.frameMask(seg.pause, spec.frames, "all");
const result = {
  metrics,
  pause: Array.from(seg.pause),
  speech: Array.from(seg.speech),
  frameCount: spec.frames,
  picks,
  frames,
  averageAll: Array.from(A.averageSpectrumDb(spec)),
  averagePause: A.averageSpectrumDb(spec, mask) ? Array.from(A.averageSpectrumDb(spec, mask)) : null,
  pauseFrames: Array.from(mask),
  runs: A.runs(seg.pause),
  energy: Array.from(seg.energy),
};
fs.writeFileSync(outPath, JSON.stringify(result));
