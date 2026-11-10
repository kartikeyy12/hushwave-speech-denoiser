"use strict";
// Usage: node wav_write.js <input.f32> <output.wav> <sample rate>
const fs = require("node:fs");
const A = require("../../demo/web/analysis.js");

const [inputPath, outputPath, rate] = process.argv.slice(2);
const raw = fs.readFileSync(inputPath);
const samples = new Float32Array(raw.buffer, raw.byteOffset, raw.length / 4);
fs.writeFileSync(outputPath, Buffer.from(A.wavBytes(samples, Number(rate))));
