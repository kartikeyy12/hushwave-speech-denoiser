"use strict";

// Frame protocol between the page and the board (docs/PROTOCOL.md).
//   A5 5A | type | seq | len (u16 LE) | payload | crc16 (LE)
// The same file runs in the browser and in Node, where the tests drive the real firmware with it.
(function (root) {
  const SYNC0 = 0xa5;
  const SYNC1 = 0x5a;
  const TYPE = {
    AUDIO_IN: 0x01,
    AUDIO_OUT: 0x81,
    HELLO: 0x02,
    HELLO_REPLY: 0x82,
    RESET: 0x03,
    RESET_ACK: 0x83,
  };
  const BLOCK = 256;
  const MAX_PAYLOAD = 1024;

  // CRC-16/CCITT-FALSE: polynomial 0x1021, initial value 0xffff, no reflection.
  const TABLE = new Uint16Array(256);
  for (let i = 0; i < 256; i++) {
    let crc = i << 8;
    for (let bit = 0; bit < 8; bit++) crc = crc & 0x8000 ? ((crc << 1) ^ 0x1021) & 0xffff : (crc << 1) & 0xffff;
    TABLE[i] = crc;
  }

  function crc16(bytes, start = 0, end = bytes.length) {
    let crc = 0xffff;
    for (let i = start; i < end; i++) crc = ((crc << 8) & 0xffff) ^ TABLE[((crc >> 8) ^ bytes[i]) & 0xff];
    return crc;
  }

  function encodeFrame(type, seq, payload) {
    const body = payload || new Uint8Array(0);
    if (body.length > MAX_PAYLOAD) throw new Error("payload too long");
    const frame = new Uint8Array(8 + body.length);
    frame[0] = SYNC0;
    frame[1] = SYNC1;
    frame[2] = type;
    frame[3] = seq & 0xff;
    frame[4] = body.length & 0xff;
    frame[5] = body.length >> 8;
    frame.set(body, 6);
    const crc = crc16(frame, 2, 6 + body.length);
    frame[6 + body.length] = crc & 0xff;
    frame[7 + body.length] = crc >> 8;
    return frame;
  }

  // Byte-stream parser. push() returns the complete, valid frames found so far.
  class FrameParser {
    constructor() {
      this.buffer = new Uint8Array(4096);
      this.length = 0;
      this.crcErrors = 0;
      this.lengthErrors = 0;
      this.skippedBytes = 0;
    }

    _drop(count) {
      this.buffer.copyWithin(0, count, this.length);
      this.length -= count;
    }

    push(chunk) {
      if (this.length + chunk.length > this.buffer.length) {
        const bigger = new Uint8Array(Math.max(this.buffer.length * 2, this.length + chunk.length));
        bigger.set(this.buffer.subarray(0, this.length));
        this.buffer = bigger;
      }
      this.buffer.set(chunk, this.length);
      this.length += chunk.length;
      const frames = [];
      for (;;) {
        let start = -1;
        for (let i = 0; i + 1 < this.length; i++) {
          if (this.buffer[i] === SYNC0 && this.buffer[i + 1] === SYNC1) {
            start = i;
            break;
          }
        }
        if (start < 0) {
          const keep = this.length > 0 && this.buffer[this.length - 1] === SYNC0 ? 1 : 0;
          this.skippedBytes += this.length - keep;
          this._drop(this.length - keep);
          return frames;
        }
        if (start > 0) {
          this.skippedBytes += start;
          this._drop(start);
        }
        if (this.length < 6) return frames;
        const type = this.buffer[2];
        const seq = this.buffer[3];
        const size = this.buffer[4] | (this.buffer[5] << 8);
        if (size > MAX_PAYLOAD) {
          this.lengthErrors++;
          this._drop(1);
          continue;
        }
        const total = 8 + size;
        if (this.length < total) return frames;
        const received = this.buffer[6 + size] | (this.buffer[7 + size] << 8);
        if (received !== crc16(this.buffer, 2, 6 + size)) {
          this.crcErrors++;
          this._drop(1);
          continue;
        }
        frames.push({ type, seq, payload: this.buffer.slice(6, 6 + size) });
        this._drop(total);
      }
    }
  }

  function pcmFromFloat(block) {
    const bytes = new Uint8Array(block.length * 2);
    const view = new DataView(bytes.buffer);
    for (let i = 0; i < block.length; i++) {
      const scaled = Math.round(block[i] * 32768);
      view.setInt16(2 * i, Math.max(-32768, Math.min(32767, scaled)), true);
    }
    return bytes;
  }

  function floatFromPcm(bytes, offset = 0, count = (bytes.length - offset) >> 1) {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const out = new Float32Array(count);
    for (let i = 0; i < count; i++) out[i] = view.getInt16(offset + 2 * i, true) / 32768;
    return out;
  }

  function decodeAudioReply(frame) {
    if (frame.type !== TYPE.AUDIO_OUT || frame.payload.length !== 4 + 2 * BLOCK) return null;
    const view = new DataView(frame.payload.buffer, frame.payload.byteOffset, frame.payload.byteLength);
    return {
      seq: frame.seq,
      computeUs: view.getUint16(0, true),
      status: view.getUint16(2, true),
      samples: floatFromPcm(frame.payload, 4, BLOCK),
    };
  }

  function decodeHello(frame) {
    if (frame.type !== TYPE.HELLO_REPLY || frame.payload.length < 7) return null;
    const view = new DataView(frame.payload.buffer, frame.payload.byteOffset, frame.payload.byteLength);
    return {
      version: view.getUint8(0),
      fs: view.getUint16(1, true),
      block: view.getUint16(3, true),
      delay: view.getUint16(5, true),
      name: String.fromCharCode(...frame.payload.subarray(7)),
    };
  }

  const api = { TYPE, BLOCK, MAX_PAYLOAD, crc16, encodeFrame, FrameParser, pcmFromFloat, floatFromPcm, decodeAudioReply, decodeHello };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.HushwaveProtocol = api;
})(typeof self !== "undefined" ? self : this);
