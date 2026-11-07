"use strict";
// Unit tests of the frame codec used by the page. Run with: node tests/js/protocol_test.js
const assert = require("node:assert/strict");
const P = require("../../demo/web/protocol.js");

const text = (s) => new TextEncoder().encode(s);

assert.equal(P.crc16(text("123456789")), 0x29b1, "CRC-16/CCITT-FALSE check value");
assert.equal(P.crc16(new Uint8Array(0)), 0xffff);

// A frame survives a round trip.
const payload = Uint8Array.from({ length: 512 }, (_, i) => (i * 7) & 0xff);
const frame = P.encodeFrame(P.TYPE.AUDIO_IN, 200, payload);
assert.equal(frame.length, 8 + 512);
assert.deepEqual([...frame.subarray(0, 6)], [0xa5, 0x5a, 0x01, 200, 0x00, 0x02]);
let parser = new P.FrameParser();
let frames = parser.push(frame);
assert.equal(frames.length, 1);
assert.equal(frames[0].type, P.TYPE.AUDIO_IN);
assert.equal(frames[0].seq, 200);
assert.deepEqual([...frames[0].payload], [...payload]);

// Byte-by-byte delivery gives the same result.
parser = new P.FrameParser();
frames = [];
for (const byte of frame) frames.push(...parser.push(Uint8Array.of(byte)));
assert.equal(frames.length, 1);

// Garbage before, between and after frames is skipped.
parser = new P.FrameParser();
const noisy = new Uint8Array([0, 1, 0xa5, 7, 9, ...frame, 0xa5, 0xa5, 3, ...frame, 0xa5]);
frames = parser.push(noisy);
assert.equal(frames.length, 2);
assert.ok(parser.skippedBytes >= 5);

// A damaged frame is dropped and the next one is found.
parser = new P.FrameParser();
const damaged = Uint8Array.from(frame);
damaged[100] ^= 0x40;
frames = parser.push(new Uint8Array([...damaged, ...frame]));
assert.equal(frames.length, 1);
assert.equal(parser.crcErrors >= 1, true);

// An impossible length is rejected without waiting for that many bytes.
parser = new P.FrameParser();
frames = parser.push(new Uint8Array([0xa5, 0x5a, 0x01, 0x01, 0xff, 0xff, ...frame]));
assert.equal(frames.length, 1);
assert.equal(parser.lengthErrors >= 1, true);

// Frames without payload.
const hello = P.encodeFrame(P.TYPE.HELLO, 3);
assert.equal(hello.length, 8);
assert.equal(new P.FrameParser().push(hello)[0].payload.length, 0);
assert.throws(() => P.encodeFrame(P.TYPE.AUDIO_IN, 0, new Uint8Array(P.MAX_PAYLOAD + 1)));

// PCM conversion: scaling, clipping and rounding match the firmware.
const pcm = P.pcmFromFloat(Float32Array.of(0, 0.5, -0.5, 1, -1, 2, -2, 1 / 32768));
const view = new DataView(pcm.buffer);
assert.deepEqual(
  [0, 1, 2, 3, 4, 5, 6, 7].map((i) => view.getInt16(2 * i, true)),
  [0, 16384, -16384, 32767, -32768, 32767, -32768, 1]
);
const back = P.floatFromPcm(pcm);
assert.equal(back[1], 0.5);
assert.equal(back[4], -1);

// Replies.
const replyPayload = new Uint8Array(4 + 2 * P.BLOCK);
new DataView(replyPayload.buffer).setUint16(0, 1234, true);
new DataView(replyPayload.buffer).setInt16(4, 16384, true);
const reply = P.decodeAudioReply({ type: P.TYPE.AUDIO_OUT, seq: 9, payload: replyPayload });
assert.equal(reply.computeUs, 1234);
assert.equal(reply.samples.length, P.BLOCK);
assert.equal(reply.samples[0], 0.5);
assert.equal(P.decodeAudioReply({ type: P.TYPE.AUDIO_OUT, seq: 1, payload: new Uint8Array(10) }), null);
const hp = Uint8Array.from([1, 0x80, 0x3e, 0x00, 0x01, 0xff, 0x01, ...text("name")]);
assert.deepEqual(P.decodeHello({ type: P.TYPE.HELLO_REPLY, seq: 1, payload: hp }), {
  version: 1,
  fs: 16000,
  block: 256,
  delay: 511,
  name: "name",
});
assert.equal(P.decodeHello({ type: P.TYPE.AUDIO_OUT, seq: 1, payload: hp }), null);
console.log("protocol: ok");
