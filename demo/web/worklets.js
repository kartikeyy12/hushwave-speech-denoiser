"use strict";

// Collects the microphone into fixed blocks and hands them to the page.
class Capture extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.size = options.processorOptions.block;
    this.buffer = new Float32Array(this.size);
    this.filled = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;
    for (let i = 0; i < channel.length; i++) {
      this.buffer[this.filled++] = channel[i];
      if (this.filled === this.size) {
        const block = this.buffer;
        this.port.postMessage(block, [block.buffer]);
        this.buffer = new Float32Array(this.size);
        this.filled = 0;
      }
    }
    return true;
  }
}

// Plays processed blocks from a ring buffer. It starts only after `prefill` samples have
// arrived, and re-primes after an underrun, so a late block costs one gap, not a stutter.
class Playback extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = options.processorOptions;
    this.prefill = opts.prefill;
    this.ring = new Float32Array(opts.capacity);
    this.read = 0;
    this.write = 0;
    this.count = 0;
    this.primed = false;
    this.underruns = 0;
    this.frames = 0;
    this.port.onmessage = (event) => {
      const data = event.data;
      for (let i = 0; i < data.length; i++) {
        if (this.count === this.ring.length) {
          this.read = (this.read + 1) % this.ring.length;
          this.count--;
        }
        this.ring[this.write] = data[i];
        this.write = (this.write + 1) % this.ring.length;
        this.count++;
      }
    };
  }

  process(inputs, outputs) {
    const out = outputs[0][0];
    if (!this.primed && this.count >= this.prefill) this.primed = true;
    if (this.primed) {
      if (this.count >= out.length) {
        for (let i = 0; i < out.length; i++) {
          out[i] = this.ring[this.read];
          this.read = (this.read + 1) % this.ring.length;
        }
        this.count -= out.length;
      } else {
        this.primed = false;
        this.underruns++;
      }
    }
    this.frames += out.length;
    if (this.frames >= 4096) {
      this.frames = 0;
      this.port.postMessage({ fill: this.count, underruns: this.underruns });
    }
    return true;
  }
}

registerProcessor("capture", Capture);
registerProcessor("playback", Playback);
