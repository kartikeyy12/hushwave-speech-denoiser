"use strict";
// Runs a whole recording through McuEngine against the real firmware with denoiseWhole, the
// code behind the Record tab. Usage: node whole_engine.js <firmware> <input.f32> <output.f32> <stats.json> [options]
const fs = require("node:fs");
const { McuEngine } = require("../../demo/web/engines.js");
const { denoiseWhole } = require("../../demo/web/analysis.js");
const { firmwarePort } = require("./fake_port.js");

const [executable, inputPath, outputPath, statsPath, optionsJson] = process.argv.slice(2);
const options = JSON.parse(optionsJson || "{}");

(async () => {
  const raw = fs.readFileSync(inputPath);
  const input = new Float32Array(raw.buffer, raw.byteOffset, raw.length / 4);
  const port = firmwarePort(executable, options);
  const engine = new McuEngine({ port, openDelayMs: 5, settleMs: 5, handshakeMs: 3000 });
  const info = await engine.open();
  const progress = [];
  const run = await denoiseWhole(engine, input, info, { onProgress: (done, total) => progress.push([done, total]) });
  await engine.close();
  fs.writeFileSync(outputPath, Buffer.from(run.output.buffer, run.output.byteOffset, run.output.byteLength));
  const finite = (values) => Array.from(values).filter((v) => Number.isFinite(v)).length;
  fs.writeFileSync(
    statsPath,
    JSON.stringify({
      info,
      blocks: run.blocks,
      received: run.received,
      lost: run.lost,
      dropped: run.dropped,
      computeCount: finite(run.computeMs),
      rttCount: finite(run.rttMs),
      lastProgress: progress[progress.length - 1],
      progressSteps: progress.length,
      length: run.output.length,
    })
  );
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
