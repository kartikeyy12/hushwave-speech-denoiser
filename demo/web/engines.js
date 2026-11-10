"use strict";

// The two places where the denoiser can run. Both take blocks of 256 samples and hand back the
// processed block, with the same interface:
//   open()               resolves {fs, block, delay, name}
//   submit(block, token) queue one block; the token comes back with the result
//   onOutput(samples, {computeMs, rttMs}, token), onError(error)
//   stats                per-block measurements, summary()
//   close()
//
// McuEngine talks to the ESP32 over USB serial (Web Serial). PythonEngine uses the server on
// this computer and exists for comparison and as a fallback when no board is connected.
(function (root) {
  const P =
    typeof module !== "undefined" && module.exports ? require("./protocol.js") : root.HushwaveProtocol;
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const MAX_SAMPLES = 60000;

  class Stats {
    constructor() {
      this.reset();
    }

    reset() {
      this.compute = [];
      this.rtt = [];
      this.dropped = 0;
      this.lost = 0;
    }

    add(computeMs, rttMs) {
      if (this.compute.length < MAX_SAMPLES) {
        this.compute.push(computeMs);
        this.rtt.push(rttMs);
      }
    }

    static describe(values) {
      if (values.length === 0) return { mean: 0, p95: 0, max: 0 };
      const sorted = Float64Array.from(values).sort();
      const sum = sorted.reduce((a, b) => a + b, 0);
      return {
        mean: sum / sorted.length,
        p95: sorted[Math.min(sorted.length - 1, Math.floor(0.95 * sorted.length))],
        max: sorted[sorted.length - 1],
      };
    }

    summary() {
      return {
        blocks: this.compute.length,
        computeMs: Stats.describe(this.compute),
        roundTripMs: Stats.describe(this.rtt),
        dropped: this.dropped,
        lost: this.lost,
      };
    }
  }

  class McuEngine {
    constructor(options = {}) {
      this.depth = options.depth || 4;
      this.maxQueue = options.maxQueue || 8;
      this.timeoutMs = options.timeoutMs || 500;
      this.baudRate = options.baudRate || 921600;
      this.handshakeMs = options.handshakeMs || 8000;
      this.openAttempts = options.openAttempts || 4;
      this.openDelayMs = options.openDelayMs === undefined ? 400 : options.openDelayMs;
      this.settleMs = options.settleMs === undefined ? 300 : options.settleMs;
      this.port = options.port || null;
      this.parser = new P.FrameParser();
      this.queue = [];
      this.inflight = new Map();
      this.seq = 0;
      this.stats = new Stats();
      this.onOutput = null;
      this.onError = null;
      this.closed = true;
      this.hello = null;
    }

    async open() {
      if (!this.port) {
        if (!(typeof navigator !== "undefined" && navigator.serial)) {
          throw new Error("This browser has no Web Serial support. Use Chrome or Edge on a computer.");
        }
        this.port = await navigator.serial.requestPort();
      }
      try {
        await this._openPort();
        await this._runState();
        if (!this.port.writable || !this.port.readable) throw new Error("The port opened but has no data streams.");
        this.writer = this.port.writable.getWriter();
        this.closed = false;
        this.readDone = this._readLoop();
        if (typeof this.port.addEventListener === "function") {
          this.port.addEventListener("disconnect", () => this._fail(new Error("The board was disconnected.")));
        }
        await this._handshake();
        await this._write(P.encodeFrame(P.TYPE.RESET, this._nextSeq()));
      } catch (error) {
        await this.close();
        throw error;
      }
      this.timer = setInterval(() => this._expire(), 100);
      return { fs: this.hello.fs, block: this.hello.block, delay: this.hello.delay, name: this.hello.name };
    }

    // The board may restart and show up again under another port while the browser opens it, so a
    // failed open is tried a few times with growing pauses.
    async _openPort() {
      let last = null;
      for (let attempt = 0; attempt < this.openAttempts; attempt++) {
        try {
          await this.port.open({ baudRate: this.baudRate, bufferSize: 1 << 16, flowControl: "none" });
          return;
        } catch (error) {
          last = error;
          if (error && (error.name === "SecurityError" || error.name === "NotAllowedError")) break;
          if (error && error.name === "InvalidStateError") await this._release();
          if (attempt + 1 < this.openAttempts) await sleep(this.openDelayMs * (attempt + 1));
        }
      }
      throw last;
    }

    // Normal run state: DTR and RTS low, then a short wait in case the board restarted on open.
    async _runState() {
      if (typeof this.port.setSignals === "function") {
        try {
          await this.port.setSignals({ dataTerminalReady: false, requestToSend: false });
        } catch (_) {
          /* not every driver supports it */
        }
      }
      await sleep(this.settleMs);
    }

    async _handshake() {
      const started = Date.now();
      while (!this.hello) {
        if (Date.now() - started > this.handshakeMs) {
          throw new Error(
            "The board did not answer. Check that the HushWave firmware is flashed and that no other program, such as the Serial Monitor, is using the port."
          );
        }
        await this._write(P.encodeFrame(P.TYPE.HELLO, this._nextSeq()));
        await sleep(250);
      }
      if (this.hello.version !== 1 || this.hello.block !== P.BLOCK) {
        throw new Error("The firmware on the board does not match this page. Flash the current firmware.");
      }
    }

    _nextSeq() {
      do {
        this.seq = (this.seq + 1) & 0xff;
      } while (this.inflight.has(this.seq));
      return this.seq;
    }

    _write(bytes) {
      return this.writer.write(bytes);
    }

    async _readLoop() {
      let ended = false;
      while (!ended && !this.closed && this.port.readable) {
        this.reader = this.port.readable.getReader();
        try {
          for (;;) {
            const { value, done } = await this.reader.read();
            if (done) {
              ended = true;
              break;
            }
            if (value) for (const frame of this.parser.push(value)) this._handle(frame);
          }
        } catch (error) {
          if (!this.closed && !this.port.readable) this._fail(error);
        } finally {
          try {
            this.reader.releaseLock();
          } catch (_) {
            /* already released */
          }
        }
      }
    }

    _handle(frame) {
      if (frame.type === P.TYPE.HELLO_REPLY) {
        const hello = P.decodeHello(frame);
        if (hello && !this.hello) this.hello = hello;
      } else if (frame.type === P.TYPE.AUDIO_OUT) {
        const reply = P.decodeAudioReply(frame);
        const entry = reply && this.inflight.get(reply.seq);
        if (!entry) return;
        this.inflight.delete(reply.seq);
        const rttMs = performance.now() - entry.t0;
        const computeMs = reply.computeUs / 1000;
        this.stats.add(computeMs, rttMs);
        if (this.onOutput) this.onOutput(reply.samples, { computeMs, rttMs }, entry.token);
        this._pump();
      }
    }

    submit(block, token) {
      if (this.closed) return;
      this.queue.push({ block, token });
      if (this.queue.length > this.maxQueue) {
        this.queue.shift();
        this.stats.dropped++;
      }
      this._pump();
    }

    _pump() {
      while (!this.closed && this.queue.length > 0 && this.inflight.size < this.depth) {
        const { block, token } = this.queue.shift();
        const seq = this._nextSeq();
        this.inflight.set(seq, { t0: performance.now(), token });
        this._write(P.encodeFrame(P.TYPE.AUDIO_IN, seq, P.pcmFromFloat(block))).catch((error) => this._fail(error));
      }
    }

    _expire() {
      const now = performance.now();
      for (const [seq, entry] of this.inflight) {
        if (now - entry.t0 > this.timeoutMs) {
          this.inflight.delete(seq);
          this.stats.lost++;
        }
      }
      this._pump();
    }

    _fail(error) {
      if (this.closed) return;
      if (this.onError) this.onError(error);
    }

    async close() {
      this.closed = true;
      clearInterval(this.timer);
      this.queue = [];
      this.inflight.clear();
      await this._release();
    }

    // Give back the streams and close the port, so that a new attempt works without a reload.
    async _release() {
      try {
        if (this.reader) await this.reader.cancel();
      } catch (_) {
        /* the stream may already be closed */
      }
      try {
        if (this.readDone) await this.readDone;
      } catch (_) {
        /* ignore */
      }
      this.reader = null;
      this.readDone = null;
      if (this.writer) {
        try {
          await this.writer.close();
        } catch (_) {
          /* ignore */
        }
        try {
          this.writer.releaseLock();
        } catch (_) {
          /* already released */
        }
        this.writer = null;
      }
      try {
        if (this.port) await this.port.close();
      } catch (_) {
        /* the port may already be closed */
      }
    }
  }

  class PythonEngine {
    constructor(options = {}) {
      this.maxQueue = options.maxQueue || 8;
      this.fetch = options.fetch || ((url, init) => fetch(url, init));
      this.queue = [];
      this.sending = false;
      this.closed = true;
      this.sid = null;
      this.stats = new Stats();
      this.onOutput = null;
      this.onError = null;
    }

    async _json(url, init) {
      const response = await this.fetch(url, init);
      if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
      return response.json();
    }

    async open() {
      const info = await this._json("/api/info");
      const session = await this._json("/api/session", { method: "POST" });
      this.sid = session.sid;
      this.closed = false;
      return { fs: info.fs, block: info.block, delay: info.delay_samples, name: "Python engine on this computer" };
    }

    submit(block, token) {
      if (this.closed) return;
      this.queue.push({ block, token });
      if (this.queue.length > this.maxQueue) {
        this.queue.shift();
        this.stats.dropped++;
      }
      this._pump();
    }

    async _pump() {
      if (this.sending) return;
      this.sending = true;
      try {
        while (!this.closed && this.queue.length > 0) {
          const { block, token } = this.queue.shift();
          const started = performance.now();
          const response = await this.fetch(`/api/process?sid=${this.sid}&enh=1`, {
            method: "POST",
            headers: { "Content-Type": "application/octet-stream" },
            body: block.buffer,
          });
          if (!response.ok) throw new Error(`engine returned HTTP ${response.status}`);
          const samples = new Float32Array(await response.arrayBuffer());
          const rttMs = performance.now() - started;
          const computeMs = Number(response.headers.get("X-Compute-Ms"));
          this.stats.add(computeMs, rttMs);
          if (!this.closed && this.onOutput) this.onOutput(samples, { computeMs, rttMs }, token);
        }
      } catch (error) {
        if (!this.closed && this.onError) this.onError(error);
      } finally {
        this.sending = false;
      }
    }

    async close() {
      const sid = this.sid;
      this.closed = true;
      this.queue = [];
      this.sid = null;
      if (sid) {
        try {
          await this.fetch(`/api/close?sid=${sid}`, { method: "POST" });
        } catch (_) {
          /* the server may already be gone */
        }
      }
    }
  }

  // Plain wording for the errors that Web Serial raises. Returns null for any other error.
  function explainSerialError(error) {
    const name = error && error.name ? error.name : "";
    const text = error && error.message ? error.message : "";
    if (/failed to open serial port/i.test(text) || name === "NetworkError") {
      return "The port could not be opened. Close the Arduino Serial Monitor and other tabs that use the board, unplug and replug the board, then try again and choose the port.";
    }
    if (/no port selected/i.test(text)) return "No port was selected. Try again and choose the board in the list.";
    if (name === "InvalidStateError" && /open/i.test(text)) {
      return "The port is already open. Reload this page and try again.";
    }
    if (name === "SecurityError") return "The browser needs a click to show the port list. Try again.";
    if (/device has been lost|disconnected/i.test(text)) {
      return "The board was disconnected. Check the cable and try again.";
    }
    return null;
  }

  const api = { Stats, McuEngine, PythonEngine, explainSerialError };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.HushwaveEngines = api;
})(typeof self !== "undefined" ? self : this);
