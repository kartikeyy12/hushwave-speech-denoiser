"use strict";
// Streams a raw float32 file through McuEngine (the page's ESP32 code) against the real firmware
// executable. Usage: node engine_roundtrip.js <firmware> <input.f32> <output.f32> <stats.json> [options]
const fs = require("node:fs");
const { McuEngine } = require("../../demo/web/engines.js");
const { firmwarePort } = require("./fake_port.js");

const [executable, inputPath, outputPath, statsPath, optionsJson] = process.argv.slice(2);
const options = JSON.parse(optionsJson || "{}");
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

(async () => {
  const raw = fs.readFileSync(inputPath);
  const input = new Float32Array(raw.buffer, raw.byteOffset, raw.length / 4);
  const blocks = input.length / 256;
  const port = firmwarePort(executable, options);
  const engine = new McuEngine({
    port,
    depth: options.depth || 4,
    handshakeMs: options.handshakeMs || 3000,
    timeoutMs: options.timeoutMs || 500,
  });
  const outputs = new Map();
  engine.onOutput = (samples, meta, token) => outputs.set(token, samples);
  const info = await engine.open();
  for (let i = 0; i < blocks; i++) {
    while (engine.queue.length + engine.inflight.size >= 6) await sleep(0);
    engine.submit(input.slice(i * 256, (i + 1) * 256), i);
  }
  const deadline = Date.now() + 10000;
  while (outputs.size + engine.stats.lost + engine.stats.dropped < blocks && Date.now() < deadline) {
    await sleep(5);
  }
  const stats = engine.stats.summary();
  await engine.close();
  const out = new Float32Array(blocks * 256);
  for (const [token, samples] of outputs) out.set(samples, token * 256);
  fs.writeFileSync(outputPath, Buffer.from(out.buffer));
  fs.writeFileSync(statsPath, JSON.stringify({ info, stats, received: outputs.size, blocks }));
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
