"use strict";
// A stand-in for a Web Serial port: the firmware executable behind two pipes.
const { spawn } = require("node:child_process");
const { Readable } = require("node:stream");

function firmwarePort(executable, options = {}) {
  const child = spawn(executable, [], { stdio: ["pipe", "pipe", "inherit"] });
  const readable = Readable.toWeb(child.stdout);
  let opened = false;
  const record = { openCalls: 0, openOptions: [], signals: [], closeCalls: 0, closed: false };
  let failuresLeft = options.openFailures || 0;
  const dropWrites = new Set(options.dropWrites || []);
  const corruptWrites = new Set(options.corruptWrites || []);
  let index = 0;
  const writable = new WritableStream({
    write(chunk) {
      const current = index++;
      if (dropWrites.has(current)) return;
      if (corruptWrites.has(current)) {
        chunk = Uint8Array.from(chunk);
        chunk[chunk.length >> 1] ^= 0x55;
      }
      return new Promise((resolve, reject) => child.stdin.write(chunk, (error) => (error ? reject(error) : resolve())));
    },
    close() {
      child.stdin.end();
    },
  });
  return {
    child,
    record,
    readable,
    writable,
    async open(openOptions) {
      record.openCalls++;
      record.openOptions.push(openOptions);
      if (failuresLeft > 0) {
        failuresLeft--;
        const error = new Error("Failed to execute 'open' on 'SerialPort': Failed to open serial port.");
        error.name = "NetworkError";
        throw error;
      }
      opened = true;
    },
    async setSignals(signals) {
      record.signals.push(signals);
    },
    // Like the browser: a port cannot be closed while one of its streams is locked.
    async close() {
      record.closeCalls++;
      if (readable.locked || writable.locked) throw new TypeError("The port has a locked stream.");
      record.closed = true;
      if (opened) child.kill();
    },
  };
}

module.exports = { firmwarePort };
