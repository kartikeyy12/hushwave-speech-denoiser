"use strict";
// A stand-in for a Web Serial port: the firmware executable behind two pipes.
const { spawn } = require("node:child_process");
const { Readable } = require("node:stream");

function firmwarePort(executable, options = {}) {
  const child = spawn(executable, [], { stdio: ["pipe", "pipe", "inherit"] });
  const readable = Readable.toWeb(child.stdout);
  let opened = false;
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
    readable,
    writable,
    async open() {
      opened = true;
    },
    async close() {
      if (opened) child.kill();
    },
  };
}

module.exports = { firmwarePort };
