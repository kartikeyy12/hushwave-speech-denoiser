"use strict";
// denoiseWhole with a stand-in engine: alignment, lost blocks, stalls, cancelling, failures.
const assert = require("node:assert/strict");
const { denoiseWhole } = require("../../demo/web/analysis.js");

const BLOCK = 256;
const DELAY = 511;

// Delays its input by DELAY samples, like the real engines do.
class FakeEngine {
  constructor(options = {}) {
    this.lose = options.lose || new Set();
    this.silent = Boolean(options.silent);
    this.stats = { lost: 0, dropped: 0 };
    this.maxQueue = 8;
    this.onOutput = null;
    this.stream = [];
    this.submitted = 0;
  }

  submit(block, token) {
    this.submitted++;
    this.stream.push(...block);
    if (this.silent) return;
    const out = new Float32Array(BLOCK);
    for (let i = 0; i < BLOCK; i++) {
      const at = token * BLOCK + i - DELAY;
      out[i] = at >= 0 ? this.stream[at] : 0;
    }
    setTimeout(() => {
      if (this.lose.has(token)) this.stats.lost++;
      else if (this.onOutput) this.onOutput(out, { computeMs: 0.5, rttMs: 1 }, token);
    }, 1);
  }
}

const info = { fs: 16000, block: BLOCK, delay: DELAY };
const samples = Float32Array.from({ length: 1000 }, (_, i) => Math.sin(i / 7) * 0.5);
const options = { intervalMs: 1 };

(async () => {
  // The output is the input again, same length, nothing missing.
  const engine = new FakeEngine();
  const previous = () => {};
  engine.onOutput = previous;
  const steps = [];
  const run = await denoiseWhole(engine, samples, info, { ...options, onProgress: (done, total) => steps.push([done, total]) });
  assert.equal(run.output.length, samples.length);
  assert.deepEqual(Array.from(run.output), Array.from(samples));
  assert.equal(run.blocks, Math.ceil((1000 + DELAY) / BLOCK));
  assert.equal(run.received, run.blocks);
  assert.equal(run.lost, 0);
  assert.equal(engine.submitted, run.blocks);
  assert.equal(engine.onOutput, previous);
  assert.deepEqual(steps[steps.length - 1], [run.blocks, run.blocks]);
  assert.ok(steps.every(([done], i) => i === 0 || done >= steps[i - 1][0]));
  assert.equal(run.computeMs.length, run.blocks);
  assert.ok(Array.from(run.computeMs).every((v) => v === 0.5));

  // A block that never comes back leaves a gap of zeros and is counted.
  const lossy = new FakeEngine({ lose: new Set([2]) });
  const gap = await denoiseWhole(lossy, samples, info, options);
  assert.equal(gap.lost, 1);
  assert.equal(gap.received, gap.blocks - 1);
  assert.ok(Number.isNaN(gap.computeMs[2]));
  assert.equal(gap.output[100], 0);
  assert.equal(gap.output[300], samples[300]);

  // An engine that stays silent ends with an error, not a hang.
  await assert.rejects(denoiseWhole(new FakeEngine({ silent: true }), samples, info, { ...options, stallMs: 150 }), /stopped answering/);

  // Cancelling.
  let cancelled = false;
  setTimeout(() => (cancelled = true), 5);
  await assert.rejects(
    denoiseWhole(new FakeEngine({ silent: true }), samples, info, { ...options, stallMs: 5000, isCancelled: () => cancelled }),
    (error) => error.name === "AbortError"
  );

  // An error reported by the engine ends the run with that error.
  const failure = new Error("The board was disconnected.");
  let fail = null;
  setTimeout(() => (fail = failure), 5);
  await assert.rejects(
    denoiseWhole(new FakeEngine({ silent: true }), samples, info, { ...options, stallMs: 5000, failed: () => fail }),
    /disconnected/
  );

  // Requests are paced: never more than the window in flight.
  const paced = new FakeEngine();
  const longer = new Float32Array(20 * BLOCK);
  await denoiseWhole(paced, longer, info, { ...options, window: 3 });
  assert.equal(paced.submitted, Math.ceil((longer.length + DELAY) / BLOCK));
  console.log("whole: ok");
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
