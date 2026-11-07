"use strict";
// Streams a raw float32 file through PythonEngine against a running demo server.
// Usage: node python_engine.js <base-url> <input.f32> <output.f32>
const fs = require("node:fs");
const { PythonEngine } = require("../../demo/web/engines.js");

const [base, inputPath, outputPath] = process.argv.slice(2);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

(async () => {
  const raw = fs.readFileSync(inputPath);
  const input = new Float32Array(raw.buffer, raw.byteOffset, raw.length / 4);
  const blocks = input.length / 256;
  const engine = new PythonEngine({ fetch: (url, init) => fetch(new URL(url, base), init) });
  const outputs = new Map();
  engine.onOutput = (samples, meta, token) => outputs.set(token, samples);
  const info = await engine.open();
  for (let i = 0; i < blocks; i++) {
    while (engine.queue.length >= 4) await sleep(0);
    engine.submit(input.slice(i * 256, (i + 1) * 256), i);
  }
  const deadline = Date.now() + 10000;
  while (outputs.size < blocks && Date.now() < deadline) await sleep(5);
  await engine.close();
  const out = new Float32Array(blocks * 256);
  for (const [token, samples] of outputs) out.set(samples, token * 256);
  fs.writeFileSync(outputPath, Buffer.from(out.buffer));
  console.log(JSON.stringify({ info, received: outputs.size, blocks }));
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
